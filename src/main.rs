mod model;

use dialoguer::Select;
use crate::model::{Layer, Network};
use std::io;

// use model;

fn main() {

    let items = vec!["Random", "File"];
    let selection = Select::new().with_prompt("Choose how to build the network").items(&items).interact().unwrap();
    // println!("You chose: {}", items[selection]);

    // for selected in selection {
    //     println!("{}", items[selected]);
    // }

    let mut input = String::new();

    if items[selection] == "Random" {
        let mut network = Network::new();
    }
    else {
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
    
        //let layer = Layer::new();
    
        network.add_layer(layer);  
    }
}
