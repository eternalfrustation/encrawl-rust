use clap::Parser;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::PathBuf;

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
    before: String,
    children: Vec<Children>,
}

#[derive(Serialize, Deserialize)]
struct Children {
    kind: String,
    data: RedditPost,
}

struct RedditClient {
    client: reqwest::blocking::Client,
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

impl RedditClient {
    fn new(client_id: String, client_secret: String) -> Result<Self, anyhow::Error> {
        let base_url = "https://www.reddit.com/";
        let client = reqwest::blocking::ClientBuilder::default().build()?;
        let req = client
            .post(format!("{base_url}api/v1/access_token"))
            .body("grant_type=client_credentials&username=&password=")
            .basic_auth(client_id.clone(), Some(client_secret.clone()))
            .header("User-Agent", "encrawl by Striking_Director_64");
        let req = req.build()?;
        let req = client.execute(req)?;
        let re = regex::Regex::new(
            r"(http|ftp|https):\\/\\/([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:\\/~+#-]*[\\w@?^=%&\\/~+#-])",
        )?;
        let req_bytes = req.bytes()?;
        let mut auth_resp: RedditAuthResp = serde_json::from_slice(&req_bytes)?;
        auth_resp.access_token = "bearer".to_owned() + &auth_resp.access_token;
        Ok(Self {
            client,
            re,
            auth_resp,
        })
    }

    fn get_posts(
        &self,
        subreddit: String,
        flairs: Vec<String>,
    ) -> Result<Vec<RedditPost>, anyhow::Error> {
        let base_url = "https://www.reddit.com";
        let request_url = format!("{base_url}/r/{subreddit}/.json");
        let mut query_param = vec![("sort", "hot")];
        let mut search_param = String::new();
        if flairs.len() != 0 {
            search_param = flairs
                .into_iter()
                .map(|flair| format!("flair:{flair}"))
                .collect::<Vec<String>>()
                .join(" OR ");
            query_param.push(("q", search_param.as_str()));
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
            .send()?;
        let resp_bytes = resp.bytes()?;
        println!("{}", String::from_utf8(resp_bytes.to_vec())?);
        let mut resp_parsed: TopLevelResp = serde_json::from_slice(&resp_bytes)?;
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

fn main() {
    let args = Args::parse();
    let reddit_client = RedditClient::new(args.token, args.secret).unwrap();
    let sub_file = BufReader::new(std::fs::File::open(args.subs).unwrap());
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
        println!(
            "{:#?}",
            reddit_client
                .get_posts(subreddit.to_string(), flairs)
                .unwrap()
        );
    }
}
