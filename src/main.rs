mod model;

use crate::model::Network;

fn main() {
    let mut network = Network::new();
    let _ = network.train(
        r"data\train-images.idx3-ubyte"
    , r"data\train-labels.idx1-ubyte", 10000);
    network.write_to_file("network_file");
}


// mod model;

// use dialoguer::Select;
// use crate::model::{Network};
// use std::io;

// // use model;

// fn main() {

//     let items = vec!["Random", "File"];
//     let selection = Select::new().with_prompt("Choose how to build the network").items(&items).interact().unwrap();

//     let mut input = String::new();
//     let mut network;

//     if items[selection] == "Random" {
//         network = Network::new();
//     }
//     else {
//         println!("Link the file you would like to build the network out of: ");
//         match io::stdin().read_line(&mut input) {
//             Ok(n) => {
//                 println!("{} bytes read", n);
//                 println!("{}", input);
//             }
//             Err(error) => println!("error: {error}"),
//         }
//         let mut network_file = String::new();
//         let n = io::stdin().read_line(&mut network_file).expect("no path given");
//         println!("{} bytes read", n);
//         println!("{}", network_file);
//         //let pattern = std::env::args().nth(1).expect("no pattern given");
//         // Sigmoid(1.0, 1.0, 2.0);
    
//         network = Network::new();
//         network.new_from_file(&network_file);
//     }
//     println!("Link a file with images: ");
//     let mut train_file_images = String::new();
//     let n = io::stdin().read_line(&mut train_file_images).expect("no path given");
//     println!("{}", train_file_images);
//     let mut train_file_labels = String::new();
//     println!("Link a file with labels: ");
//     let m = io::stdin().read_line(&mut train_file_labels.trim()).expect("no path given");
//     let _ = network.train(&train_file_images, &train_file_labels, 10000);
//     network.write_to_file("network_file");
// }