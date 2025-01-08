mod model;
// mod graphics;

// use libloading::Library;
// use std::{thread::sleep, time::Duration};
// use tokio::time::{sleep, Duration};

#[allow(unused_imports)]
use model::{normalize_output, Layer, ActivationFunction, Network, ClassicNetwork, ComputeNetwork};
use rand::Rng;
use renderdoc::{RenderDoc, V110};

#[tokio::main]
async fn main() {
    
    // let renderdoc_path = "/opt/renderdoc/lib/librenderdoc.so"; // Update with the correct path
    // unsafe {let lib = Library::new(renderdoc_path).expect("Failed to load RenderDoc library");}
    // println!("RenderDoc library loaded successfully!");
    // std::env::set_var("LD_LIBRARY_PATH", renderdoc_path);

    // let mut rd: RenderDoc<V110> = RenderDoc::new().expect("Unable to initialize RenderDoc");

    // rd.start_frame_capture(std::ptr::null(), std::ptr::null());

    // tokio::time::sleep(Duration::from_secs(1)).await;
    // let file_name = "data/training-data/archive/...";

    // let mut network: ClassicNetwork = Network::new();
    let mut network: ComputeNetwork = Network::new_default();
    // let mut network: ClassicNetwork = Network::new_default();

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

    // let input = vec![0.0; 28*28];
    let mut rng = rand::thread_rng();
    let input: Vec<f32> = (0..28*28)
        .map(|_| rng.gen_range(0.0..1.0)).collect();
    println!("Input: {:?}", input);
    let raw_result = network.calculate(&input).await;

    // println!("Raw Output: {:?}", raw_result);
    // let cooked_result = normalize_output(raw_result.last().unwrap());
    // let cooked_result = normalize_output(&raw_result);

    // println!("Output: {:?}", cooked_result);
    println!("Output: {:?}", raw_result);

    // rd.trigger_capture();

    // rd.end_frame_capture(std::ptr::null(), std::ptr::null());

    // sleep(Duration::from_secs(5)).await;
}
