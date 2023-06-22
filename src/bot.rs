use dotenv::dotenv;
use femto_gpt::gpt::{TrainingState, GPT};
use femto_gpt::graph::gpu::GpuGraph;
use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};
use poise::serenity_prelude as serenity;
use std::fs;
use std::io::Read;
use std::path::Path;
use tokio::sync::Mutex;

struct Data {
    gpt: Mutex<GPT<GpuGraph>>,
    tokenizer: SimpleTokenizer,
}

fn load_gpt() -> Result<Data, Error> {
    // load GPT
    let graph = femto_gpt::graph::gpu::GpuGraph::new()?;
    let is_gpu = true;

    let training_state_path = Path::new("training_state.dat");

    let mut rng = rand::thread_rng();

    // Create a unique char-to-int mapping for all unique characters inside our dataset
    let dataset_char = fs::read_to_string("dataset.txt")?;
    let tokenizer = SimpleTokenizer::new(&dataset_char);

    let batch_size = 32;
    let num_tokens = 64;
    let vocab_size = tokenizer.vocab_size();
    let embedding_degree = 64;
    let num_layers = 4;
    let num_heads = 4;
    let head_size = embedding_degree / num_heads;
    let dropout = 0.0;

    assert_eq!(num_heads * head_size, embedding_degree);

    println!("Vocab-size: {} unique characters", vocab_size);

    let mut gpt = GPT::new(
        &mut rng,
        graph,
        is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
        vocab_size,
        embedding_degree,
        num_tokens,
        num_layers,
        num_heads,
        head_size,
        dropout,
    )?;

    gpt.sync()?;

    if training_state_path.is_file() {
        let mut ts_file = fs::File::open(training_state_path).unwrap();
        let mut bytes = Vec::new();
        ts_file.read_to_end(&mut bytes).unwrap();
        let ts: TrainingState = bincode::deserialize(&bytes).unwrap();
        gpt.set_training_state(ts, false)?;
    }

    Ok(Data {
        gpt: Mutex::new(gpt),
        tokenizer,
    })
}

type Error = Box<dyn std::error::Error + Send + Sync>;
type Context<'a> = poise::Context<'a, Data, Error>;

#[poise::command(slash_command)]
async fn infer(ctx: Context<'_>) -> Result<(), Error> {
    let inference_temperature = 0.5;

    let data = ctx.data();

    let mut gpt = data.gpt.lock().await;
    let tokenizer = &data.tokenizer;

    let inference = gpt.infer(
        &mut rand::thread_rng(),
        &tokenizer.tokenize("\n"),
        100,
        inference_temperature,
        |_| {},
    )?;

    let inference_text = tokenizer.untokenize(&inference);

    ctx.say(inference_text).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenv().unwrap();

    // create bot
    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![infer()],
            ..Default::default()
        })
        .token(std::env::var("DISCORD_TOKEN").expect("missing DISCORD_TOKEN"))
        .intents(
            serenity::GatewayIntents::non_privileged() | serenity::GatewayIntents::MESSAGE_CONTENT,
        )
        .setup(|ctx, _ready, framework| {
            Box::pin(async move {
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                Ok(load_gpt()?)
            })
        });

    framework.run().await?;

    Ok(())
}
