use sqlx::FromRow;
use clap::Parser;
use encrawl_rust::mamba::{init, TextGeneration};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres};
use std::io::prelude::*;
use std::io::BufReader;
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

#[derive(Debug, Serialize, Deserialize, FromRow)]
struct Article {
    title: String,
    url: String,
    content: String,
    author: String,
}

impl Article {
    fn get_embedding(&self, model: &SentenceEmbeddingsModel) -> anyhow::Result<Vec<f32>> {
        Ok(model.encode(&[self.title.clone()])?[0].clone())
    }

    async fn store(
        &self,
        db: Arc<Pool<sqlx::Postgres>>,
        model: &SentenceEmbeddingsModel,
    ) -> anyhow::Result<()> {
        let embedding = self.get_embedding(model)?;
        sqlx::query("INSERT INTO articles (title, url, content, author, embedding) VALUES ($1, $2, $3, $4, $5)")
            .bind(self.title.clone())
            .bind(self.url.clone())
            .bind(self.content.clone())
            .bind(self.author.clone())
            .bind(pgvector::Vector::from(embedding))
            .execute(db.as_ref()).await?;
        Ok(())
    }
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
async fn search(
    db: Arc<Pool<Postgres>>,
    model: &SentenceEmbeddingsModel,
    query: String,
    limit: i32,
) -> anyhow::Result<Vec<Article>> {
    let embedding = pgvector::Vector::from(model.encode(&[query])?[0].clone());
    Ok(sqlx::query_as::<_, Article>(
        "SELECT title, content, url, author FROM articles ORDER BY embedding <=> $1 LIMIT $2",
    )
    .bind(embedding)
    .bind(limit)
    .fetch_all(db.as_ref())
    .await?)
}

trait Summarisable {
    fn get_summary(&self, text_generator: &mut TextGeneration) -> anyhow::Result<String>;
}

impl Summarisable for Vec<Article> {
    fn get_summary(&self, text_generator: &mut TextGeneration) -> anyhow::Result<String> {
        let prompt = String::from("You are an conversational AI model designed to create summaries of news given to you on a specific topic. Do NOT use lists, Just output in paragraphs in Markdown.")
        + &self.into_iter()
            .enumerate()
            .map(|(i,a)| 
                format!("Article: {i}\nTitle: {}\nAuthor: {}\nUrl: {}\nContent: {}\n",
                    a.title,
                    a.author,
                    a.url,
                    a.content)).collect::<Vec<String>>()
            .join("\n")
        +  "User: Summarize the given news. You MUST add the relevant links to the content using markdown links in the format of [<Title>](<Url>).\nResponse: ";
        text_generator.run(&prompt, 200)
    }
}

fn main() -> anyhow::Result<()> {
    colog::init();
    let args = Args::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let pool = rt.block_on(
        PgPoolOptions::new()
            .max_connections(5)
            .connect("postgres://postgres:123456789@localhost/encrawl"),
    )?;
    rt.block_on(sqlx::query("CREATE EXTENSION IF NOT EXISTS vector").execute(&pool))?;
    let pool = Arc::new(pool);
    let reddit_client = rt.block_on(RedditClient::new(args.token, args.secret))?;
    let sub_file = BufReader::new(std::fs::File::open(args.subs).unwrap());
    let scrapers = ScraperConfig::from_file(args.scraper).unwrap();
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;
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
        for article in rt
            .block_on(reddit_client.get_posts(subreddit.to_string(), flairs))?
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
            rt.block_on(async {
                match article.await.unwrap().store(pool.clone(), &model).await {
                    Ok(_) => {}
                    Err(e) => log::error!("{}", e),
                }
            })
        }
    }
    println!(
        "{:#?}",
        rt.block_on(search(pool, &model, String::from("Tax"), 4))
            .unwrap().get_summary(&mut init()?)
    );
    Ok(())
}
