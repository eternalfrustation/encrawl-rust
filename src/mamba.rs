#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use candle_transformers::models::mamba::{Config, Model, State};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct TextGeneration {
    model: Model,
    config: Config,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        config: Config,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            config,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        use std::io::Write;
        let dtype = self.model.dtype();
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let mut generated_tokens = 0usize;
        let binding = self.tokenizer.get_vocab(true);
        let eos_token = match binding.get("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let mut state = State::new(1, &self.config, dtype, &self.device)?;
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[t], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;
            next_logits = Some(logits);
        }

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
            let logits = logits.squeeze(0)?.to_dtype(dtype)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == *eos_token {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?)
        }
        let dt = start_gen.elapsed();
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(self
            .tokenizer
            .decode(tokens.as_slice(), true)
            .map_err(|_| std::fmt::Error::default())?)
    }
}

#[derive(Parser, ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    Mamba2_8bSlimPj,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            Self::Mamba2_8bSlimPj => "state-spaces/mamba-2.8b-slimpj",
        }
    }

    fn revision(&self) -> &'static str {
        match self {
            Self::Mamba130m
            | Self::Mamba370m
            | Self::Mamba790m
            | Self::Mamba1_4b
            | Self::Mamba2_8bSlimPj => "refs/pr/1",
            Self::Mamba2_8b => "refs/pr/4",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long, default_value = "mamba130m")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

pub fn init() -> Result<TextGeneration> {
    use std::str::FromStr;


    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        Which::Mamba2_8bSlimPj.model_id().to_string(),
        RepoType::Model,
        Which::Mamba2_8bSlimPj.revision().to_string(),
    ));
    let tokenizer_filename = api
        .model("EleutherAI/gpt-neox-20b".to_string())
        .get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let filenames = repo.get("model.safetensors")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let device = Device::Cpu;
    let dtype = DType::from_str("f32")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&vec![filenames], dtype, &device)? };
    let model = Model::new(&config, vb.pp("backbone"))?;

    Ok(TextGeneration::new(
        model,
        config,
        tokenizer,
        299792458,
        None,
        None,
        1.1,
        64,
        &device,
    ))
}


