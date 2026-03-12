fn main() {
    match langchain_model_profiles::cli::run(std::env::args().skip(1)) {
        Ok(output) => {
            println!("{output}");
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(error.exit_code());
        }
    }
}
