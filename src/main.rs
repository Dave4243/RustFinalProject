mod model;
// mod graphics;

#[allow(unused_imports)]
use model::{normalize_output, Layer, ActivationFunction, Network, ClassicNetwork, ComputeNetwork};


#[tokio::main]
async fn main() {

    // let file_name = "data/training-data/archive/...";

    // let mut network: ClassicNetwork = Network::new();
    // let mut network: ComputeNetwork = Network::new_default();
    let mut network: ClassicNetwork = Network::new_default();

    // let weights: Vec<Vec<f32>> = vec![vec![0.0; 28*28]; 28*28];
    // let mut layer = Layer::new(28*28, weights);
    // layer.biases = None;
    // network.add_layer(layer.clone()).unwrap();    
    // network.add_layer(layer).unwrap();    

    // let weights: Vec<Vec<f32>> = vec![vec![0.0; 28*28]; 10];
    // let mut output_layer = Layer::new(10, weights);
    // output_layer.biases = None;
    // if let Err(x) = network.add_layer(output_layer) {
    //     panic!("Failed to add layer! problem: {:?}", x);
    // }
    // println!("first layer: {}  |  last layer: {}", network.input_layer().unwrap().size, network.output_layer().unwrap().size);
    
    // network.set_output_activation(&ActivationFunction::Sigmoid);
    // network.set_hidden_activation(&ActivationFunction::Tanh);

    let input = vec![0.0; 28*28];
    let raw_result = network.calculate(&input).await;

    println!("Raw Output: {:?}", raw_result);
    // let cooked_result = normalize_output(raw_result.last().unwrap());
    let cooked_result = normalize_output(&raw_result);

    println!("Output: {:?}", cooked_result);


    // println!("Hello, world!");
}
