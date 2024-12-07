mod model;

use dialoguer::MultiSelect;
use crate::model::{Layer, Sigmoid, Network};
use std::io;

// use model;

fn main() {

    let items = vec!["Random", "File"];
    let selection = MultiSelect::new().with_prompt("Choose a mode").items(&items).interact().unwrap();
    println!("You chose: ");

    for i in selection {
        println!("{}", items[i]);
    }

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(n) => {
            println!("{} bytes read", n);
            println!("{}", input);
        }
        Err(error) => println!("error: {error}"),
    }
    let path = std::env::args().nth(1).expect("no path given");
    let pattern = std::env::args().nth(1).expect("no pattern given");
    // Sigmoid(1.0, 1.0, 2.0);

    let file_name = "";

    let mut network = Network::new();

    let layer = Layer::new();

    network.add_layer(layer);    

    // println!("Hello, world!");
}
