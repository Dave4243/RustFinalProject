mod model;
mod graphics;

use crate::model::{Layer, ActivationFunction, Network};

// use model;

fn main() {
    // Sigmoid(1.0, 1.0, 2.0);

    // let file_name = "data/training-data/archive/...";

    let mut network = Network::new();

    let weights: Vec<Vec<f64>> = vec![vec![0.0; 28*28]; 28*28];
    let mut layer = Layer::new(28*28, weights);
    layer.biases = None;
    network.add_layer(layer.clone()).unwrap();    
    network.add_layer(layer).unwrap();    

    let weights: Vec<Vec<f64>> = vec![vec![0.0; 28*28]; 10];
    let mut output_layer = Layer::new(10, weights);
    output_layer.biases = None;
    if let Err(x) = network.add_layer(output_layer) {
        panic!("Failed to add layer! problem: {:?}", x);
    }
    // println!("first layer: {}  |  last layer: {}", network.input_layer().unwrap().size, network.output_layer().unwrap().size);
    
    network.set_output_activation(&ActivationFunction::Sigmoid);
    network.set_hidden_activation(&ActivationFunction::Tanh);

    let input = vec![0.0; 28*28];
    let result = network.calculate(input);

    println!("Output: {:?}", result);


    // println!("Hello, world!");
}
