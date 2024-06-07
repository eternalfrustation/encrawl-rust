use clap::Parser;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
struct ScraperConfig {
    domain: String,
    author_selector: String,
    content_selector: String,
    title_selector: String,
}

impl ScraperConfig {
    fn from_file(path: PathBuf) -> anyhow::Result<Vec<Self>> {
        Ok(ron::from_str(&String::from_utf8(std::fs::read(path)?)?)?)
    }

    async fn get_article(&self, url: String) -> anyhow::Result<Article> {
        let document =
            scraper::Html::parse_document(&reqwest::get(url.clone()).await?.text().await?);
        let author_selector = scraper::Selector::parse(&self.author_selector).unwrap();
        let content_selector = scraper::Selector::parse(&self.content_selector).unwrap();
        let title_selector = scraper::Selector::parse(&self.title_selector).unwrap();
        let author = document
            .select(&author_selector)
            .map(|e| e.text().to_owned().collect::<Vec<&str>>().join("\n"))
            .collect::<Vec<String>>()
            .join("\n");
        let content = document
            .select(&content_selector)
            .map(|e| e.text().to_owned().collect::<Vec<&str>>().join("\n"))
            .collect::<Vec<String>>()
            .join("\n");
        let title = document
            .select(&title_selector)
            .map(|e| e.text().to_owned().collect::<Vec<&str>>().join("\n"))
            .collect::<Vec<String>>()
            .join("\n");
        Ok(Article {
            title,
            author,
            content,
            url,
        })
    }
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    token: String,

    #[arg(short, long)]
    secret: String,

    #[arg(long, default_value = (PathBuf::from("finance_subs.list")).into_os_string())]
    subs: PathBuf,

    #[arg(long, default_value = (PathBuf::from("scrapers.ron")).into_os_string())]
    scraper: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct TopLevelResp {
    kind: String,
    data: TopLevelData,
}

#[derive(Serialize, Deserialize)]
struct TopLevelData {
    after: String,
    dist: isize,
    modhash: String,
    before: Option<String>,
    children: Vec<Children>,
}

#[derive(Serialize, Deserialize)]
struct Children {
    kind: String,
    data: RedditPost,
}

struct RedditClient {
    client: reqwest::Client,
    re: regex::Regex,
    auth_resp: RedditAuthResp,
}

#[derive(Serialize, Deserialize)]
struct RedditAuthResp {
    access_token: String,
    token_type: String,
    expires_in: i64,
    scope: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RedditPost {
    title: String,
    url: String,
    selftext: String,
    over_18: bool,
    stickied: bool,
    body: Option<String>,
    #[serde(skip_deserializing)]
    referenced_url: String,
}

#[derive(Debug)]
struct Article {
    title: String,
    url: String,
    content: String,
    author: String,
}

impl RedditClient {
    async fn new(client_id: String, client_secret: String) -> Result<Self, anyhow::Error> {
        let base_url = "https://www.reddit.com/";
        let client = reqwest::ClientBuilder::default().build()?;
        let req = client
            .post(format!("{base_url}api/v1/access_token"))
            .body("grant_type=client_credentials&username=&password=")
            .basic_auth(client_id.clone(), Some(client_secret.clone()))
            .header("User-Agent", "encrawl by Striking_Director_64");
        let req = req.build()?;
        let req = client.execute(req).await?;
        let re = regex::Regex::new(
            r"(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])",
        )?;
        let mut auth_resp: RedditAuthResp = serde_json::from_slice(&req.bytes().await?)?;
        auth_resp.access_token = "bearer".to_owned() + &auth_resp.access_token;
        Ok(Self {
            client,
            re,
            auth_resp,
        })
    }

    async fn get_posts(
        &self,
        subreddit: String,
        flairs: Vec<String>,
    ) -> Result<Vec<RedditPost>, anyhow::Error> {
        let base_url = "https://www.reddit.com";
        let request_url = format!("{base_url}/r/{subreddit}/.json");
        let mut query_param = vec![("sort", "hot")];
        let search_param = if flairs.len() == 0 {
            None
        } else {
            Some(
                flairs
                    .into_iter()
                    .map(|flair| format!("flair:{flair}"))
                    .collect::<Vec<String>>()
                    .join(" OR "),
            )
        };
        match &search_param {
            None => {}
            Some(search_param) => {
                query_param.push(("q", search_param.as_str()));
            }
        }
        let resp = self
            .client
            .get(request_url)
            .header("Authorization", self.auth_resp.access_token.clone())
            .header(
                "User-Agent",
                "telegram-integration-bot by Striking_Director_64",
            )
            .query(&query_param)
            .send()
            .await?;
        let mut resp_parsed: TopLevelResp = serde_json::from_slice(&resp.bytes().await?)?;
        Ok(resp_parsed
            .data
            .children
            .iter_mut()
            .map(|p| &mut p.data)
            .map(|post| {
                match self.re.find(&post.selftext.clone()) {
                    Some(url) => post.referenced_url = url.as_str().to_string(),
                    None => match &post.body {
                        Some(body) => match self.re.find(&body.clone()) {
                            Some(url) => post.referenced_url = url.as_str().to_string(),
                            None => {}
                        },
                        None => {}
                    },
                };
                post.clone()
            })
            .collect())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    colog::init();
    let args = Args::parse();
    let reddit_client = RedditClient::new(args.token, args.secret).await?;
    let sub_file = BufReader::new(std::fs::File::open(args.subs).unwrap());
    let scrapers = ScraperConfig::from_file(args.scraper).unwrap();
    for line in sub_file.lines().flatten() {
        let mut line = line.split_ascii_whitespace();
        let subreddit = match line.next() {
            Some(sub) => sub,
            None => continue,
        };
        let flairs = match line.next() {
            Some(flairs) => flairs
                .split(',')
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            None => vec![],
        };
        for url in reddit_client
            .get_posts(subreddit.to_string(), flairs)
            .await?
            .into_iter()
            .map(|post| post.url)
            .filter(|url| !url.contains("reddit.com") && !url.contains("redd.it"))
            .map(|url| {
                match scrapers
                    .iter()
                    .filter(|scraper| url.contains(&scraper.domain))
                    .next()
                {
                    Some(scraper) => Some((url, scraper)),
                    None => {
                        log::warn!("Scraper for {} not found", url);
                        None
                    }
                }
            })
            .flatten()
            .map(|(url, scraper)| scraper.get_article(url))
        {
            println!("{:?}", url.await.unwrap())
        }
    }
    Ok(())
}
