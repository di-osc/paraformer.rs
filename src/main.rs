use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    name: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve the application
    Serve(ServeArgs),
}

#[derive(Args)]
struct ServeArgs {
    /// The host to bind to for serving.
    #[arg(short, long, default_value = "0.0.0.0")]
    host: String,
    /// The port to bind to for serving.
    #[arg(short, long, default_value = "8080")]
    port: u16,
}

fn main() {
    let _cli = Cli::parse();
}
