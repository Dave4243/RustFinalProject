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
    let mut network;

    if items[selection] == "Random" {
        network = Network::new();
    }
    else {
        println!("Link the file you would like to build the network out of: ");
        match io::stdin().read_line(&mut input) {
            Ok(n) => {
                println!("{} bytes read", n);
                println!("{}", input);
            }
            Err(error) => println!("error: {error}"),
        }
        let mut network_file = String::new();
        let n = io::stdin().read_line(&mut network_file).expect("no path given");
        println!("{} bytes read", n);
        println!("{}", network_file);
        //let pattern = std::env::args().nth(1).expect("no pattern given");
        // Sigmoid(1.0, 1.0, 2.0);
    
        network = Network::new();
        network.new_from_file(&network_file);

        //network.calculate()    
    
        //let layer = Layer::new();
    
        //network.add_layer(layer);  
    }
    println!("Link the file you would like to train the model on: ");
    let mut train_file = String::new();
    let n = io::stdin().read_line(&mut train_file).expect("no path given");
    println!("{} bytes read", n);
    println!("{}", train_file);
}
