#![allow(dead_code)]

use itertools::{interleave, Itertools};
use std::{f32::consts::E, num::NonZero};
use std::vec::Vec;

use rust_webgpu::{start_logger, Compute, State};
use rand::Rng;

const IMG_WIDTH: usize = 28;
const IMG_HEIGHT: usize = 28;

#[derive(Copy, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    Softmax,
    None,
}
impl ActivationFunction {
    pub fn calculate(self, input: f32) -> f32 {
        match self {
            Self::Sigmoid => {
                1.0 / (1.0 + (-input).exp())
            },
            Self::Tanh => {
                (E.powf(input)-E.powf(-input)) / (E.powf(input) + E.powf(-input))
            },
            Self::Relu => input.max(0.0),
            Self::Softmax => input, // needs input vector, so this is handled in function
            Self::None => input,
        }
    }
}

/// Describes 1D vector layer. 
/// 
/// Each layer contains the weights that relate it to its previous layer.
/// 
/// The matrix multiplation of the previous layer's output and this layer's weights matrix results in this layers output 
/// 
/// Also contains optional biases, there is a bias for each node in the layer. This repesents a flat amount to add to the activation sum after accounting for weights
#[derive(Clone)]
pub struct Layer {
    size: usize,

    // The outer vec represents the current layer's neurons
    // the inner vec represents the previous layer's neurons
    // CLARIFY: This is worded confusingly. 
    // It seems you mean that the rows of the weights matrix are 
    // dotted with the input to the layer to produce each row of output
    // The input vector is in column major format, to be clear.
    weights: Vec<Vec<f32>>,

    // The bias for each neuron. Should be added to the dot product of 
    // previous weights and current weights
    // CLARIFY: If firstly, weights are a matrix, so it should be the matrix product not dot product.
    // Secondly, the weights of the previous layers are not used in the running calculation. The weights matrix represents
    // the relationship between this layer and the previous
    biases: Vec<f32>,

    // The input from the previous layer ("x")
    input : Vec<f32>,

    // Values z = W*x + b
    // This represents the output before the activation function is applied
    z_values : Vec<f32>,

    output : Vec<f32>,

    activation_function: ActivationFunction,
}

impl Layer { 
    /// initalizes random weights and biases, 
    /// input size is the size of the previous layer
    /// output size is the size of this layer (how many neurons in the layer)
    fn new_random(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        
        // Randomly initialize weights and biases
        let weights: Vec<Vec<f32>> = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();

        let biases: Vec<f32> = (0..output_size).map(|_| rng.gen_range(-0.5..0.5)).collect();

        Self {
            // size : input_size,
            size : output_size,
            weights,
            biases,
            input : vec![],
            z_values : vec![],
            output: vec![],
            activation_function : activation,
        }
    }

    pub fn calculate(self: &mut Self, prev_layer_out: &Vec<f32>) -> Vec<f32> {
        assert!(self.weights.len() == self.size, "Number of rows (length of columns) in weight, {}, does not match output layer size, {}!", self.weights.len(), self.size);
        self.input = prev_layer_out.clone();

        // Input vector is column major.
        let mut result: Vec<f32> = vec![0.0; self.size]; 

        // Col is the index of the layer vector
        for (col, weights_row) in self.weights.iter().enumerate() {
            let mut sum: f32 = weights_row.iter()
                .zip(prev_layer_out)
                .map(|(x, y)| x*y)
                .sum();
            sum += self.biases[col];
            result[col] = sum;
        }
        // Store the pre-activation values
        self.z_values = result.clone();
        
        // we see a problem arise if "x" is too large or too small (saturated network)
        // because the sigmoid will map it to -1 or 1, and the gradient
        // may vanish as a result. There are techniques to combat this issue
        // but we've initalized weights and biases to [-0.5, 0.5] for now.
        match self.activation_function {
            ActivationFunction::Softmax => {
                // Find the maximum value for numerical stability
                let max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = result.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                result = exps.iter().map(|&x| x / sum).collect();
            },
            _ => result.iter_mut().for_each(|x| *x = self.activation_function.calculate(*x)),
        }
        // Store results and return
        self.output = result.clone();
        return result;
    }
    
    fn backpropagate(&mut self, output_gradient: Vec<f32>, learning_rate: f32) -> Vec<f32> {
        let dLdz : Vec<f32>;
        match self.activation_function {
            ActivationFunction::Relu => {
                dLdz = output_gradient.iter()
                .zip(self.z_values.iter())
                .map(|(&og, &iv)| if iv > 0.0 { og } else { 0.0 })
                .collect();
            },
            ActivationFunction::Softmax => dLdz = output_gradient,
            _ => return Vec::new(),
        }

        // update weights
        // dL/dW = dL/dz * dz/dW = dL/dz * input
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                let dLdW = dLdz[i] * self.input[j];
                self.weights[i][j] -= learning_rate * dLdW;
            }
        }

        // update biases, dL/db = dL/dz * dz/dB = dL/dz * 1 = dL/dz
        for i in 0..self.biases.len() {
            self.biases[i] -= learning_rate * dLdz[i];
        }

        // computes gradient with respect to input (output of prev layer)
        let mut dLdx = vec![0.0; self.input.len()];
        for j in 0..self.input.len() {
            for i in 0..self.weights.len() {
                dLdx[j] += dLdz[i] * self.weights[i][j];
            }
        }
        // pass this gradient back
        return dLdx;
    }
}



// enum NetworkDim {
//     Invalid = -1,
//     Network2D = 2,
//     Network3D = 3,
// }



// #[derive(Clone)]
pub struct ComputeNetwork<'a> {
    layers: Vec<Layer>,
    learning_rate : f32,
    // Optional since lazy initialized
    gpu_state: Option<State<'a>>,
    compute: Option<Compute<f32>>,
    input : Vec<f32>,
    output: Vec<f32>,
}

#[repr(C)]
// #[derive(Copy, Clone, Debug, /*bytemuck::Pod, bytemuck::Zeroable*/)]
struct BufferData {
    // prev_layer_size: u32,
    weights_dims: [u32;2],
    curr_layer_index: u32,
    // Contains size of every layer, including input layer at index 0
    // This means it is not one to one with the networks layers vector!
    output_layer_sizes_and_activations: Vec<u32>
}
impl BufferData {
    pub fn new_from_layer(input_layer: &Vec<f32>, layers: &Vec<Layer>, index: u32) -> Self {
        let weights_dims: [u32; 2] = (layers.iter().flat_map(|layer| layer.weights.iter().map(|row| row.len() as u32)).max().unwrap(), layers.iter().map(|i| i.weights.len() as u32).max().unwrap()).into();
        
        let mut layer_sizes: Vec<u32> = Vec::new();
        layer_sizes.push(input_layer.len() as u32);
        layers.iter().for_each(|i| layer_sizes.push(i.size as u32));
        
        let mut layer_activations: Vec<u32> = Vec::new();
        layer_activations.push(input_layer.len() as u32);
        layers.iter().for_each(|i| layer_sizes.push(i.activation_function as u32));

        return BufferData{
            weights_dims,
            curr_layer_index: index,
            output_layer_sizes_and_activations: layer_sizes
                .into_iter()
                .interleave(layer_activations.into_iter())
                .collect::<Vec<_>>()
        };
    }

    pub fn to_bytes(self:&Self) -> Vec<u8> {
        let buffer_data_values = [
            self.weights_dims[0],
            self.weights_dims[1],
            self.curr_layer_index,
        ];
        let buffer_data_bytes: &[u8] = bytemuck::cast_slice(&buffer_data_values);
        let output_layer_sizes_and_activations_bytes: &[u8] = bytemuck::cast_slice(&self.output_layer_sizes_and_activations);
        let combined_buffer_data_bytes: Vec<u8> = buffer_data_bytes.iter().chain(output_layer_sizes_and_activations_bytes.iter()).copied().collect();
        
        return combined_buffer_data_bytes;
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OutputData {
    z_value: f32,
    layer_output: f32
}

impl Network for ComputeNetwork<'static> {

    fn get_layers(self: &Self) -> &Vec<Layer> {
        return &self.layers;
    }
    fn get_layers_mut(self: &mut Self) -> &mut Vec<Layer> {
        return &mut self.layers;
    }

    fn new_default() -> Self { 
        let mut network = Self {
            layers : vec![],
            learning_rate : 0.1,
            input : vec![],
            output : vec![],
            // State and compute are lazy initialized, they cannot be sized until layers is complete at caluclation time
            gpu_state: None,
            compute: None, 
        };  

        // single layer neural network
        let first_layer = Layer::new_random
        (784, 400, ActivationFunction::Relu);
        let output_layer = Layer::new_random
        (400, 10, ActivationFunction::Softmax);
        network.layers.push(first_layer);
        network.layers.push(output_layer);
        return network;
    }

    fn new(learning_rate: f32) -> Self {
        Self {
            layers : vec![],
            learning_rate,
            input : vec![],
            output : vec![],
            // State and compute are lazy initialized, they cannot be sized until layers is complete at caluclation time
            gpu_state: None,
            compute: None, 
        }
    }

    /// Steps:
    /// 1. find size of largest layer, in order to size output buffer and prev_layer buffer
    /// 1. find size of largest weights matrix, in order to size input buffer
    /// 1.5 Optimal number of threads = output_buffer_size
    /// 2. generate buffers
    /// 3. Move input layer into output buffer
    /// 4. Move weights matrix into input_buffer
    /// 5. Calculate (matrix multiply input and output, put results into)
    /// CHANGES:
    /// 1. Required to update layer input and output properties individually (does not call layer.calculate!)
    /// 2. Must return only needed outputs, not full output of all layers 
    async fn calculate(self: &mut Self, input: &Vec<f32>) -> Vec<f32>{
        start_logger();
    
        // let event_loop = EventLoop::new().unwrap();

        // Creates GPU pipeline
        #[allow(unused_mut)]
        if self.gpu_state.is_none() {
            self.gpu_state = Some(State::new_no_graphics().await);
        }
        let state: &State = self.gpu_state.as_ref().unwrap();

        let mut max_layer_size = input.len();
        for layer in self.layers.iter() {
            if layer.size > max_layer_size {max_layer_size = layer.size;}
        }

        let mut max_weight_size = 0;
        for layer in self.layers.iter() {
            let weight_size = layer.weights.iter().flatten().count();
            if weight_size > max_weight_size {max_weight_size = weight_size;}
        }

        let buffer_data_buffer_size = size_of::<BufferData>();

        let mut input_sizes: Vec<NonZero<u32>> = Vec::new(); //Vec::from_iter((0..2).map(|_| NonZero::new(16).unwrap()));      
        input_sizes.push(NonZero::new(max_layer_size as u32).unwrap());
        input_sizes.push(NonZero::new(max_weight_size as u32).unwrap());
        input_sizes.push(NonZero::new(buffer_data_buffer_size as u32).unwrap());


        let mut layer_sizes: Vec<u32> = Vec::new();
        layer_sizes.push(input.len() as u32);
        self.layers.iter().for_each(|i| layer_sizes.push(i.size as u32));
        // This has to be the sum of the sizes of every layer, including the input, since that one isnt caluclated by the shader.
        // let output_size = NonZero::new(self._layers.iter().map(|i| i.size as u32).sum()).unwrap();
        let output_size: u32 = &layer_sizes.iter().sum() * 2u32;

        let shader_str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/matrix.wgsl"));        
        let mut compute: Compute<f32> = Compute::new(&state.device, shader_str, input_sizes, output_size.try_into().unwrap());

        // Set to calculate layer 1 (layer 1 being the one after the input layer, this is the same as _layers[0]) 
        // let buffer_data = BufferData::new_from_layer(&input, &self._layers, 1);
        // let buffer_data_bytes: Vec<u8> =  buffer_data.to_bytes();

        // let cooked_input: Vec<OutputData> = input.chunks(2).map(|c| OutputData{z_value: c.0, layer_output: c.1}).collect();
        let cooked_input: Vec<OutputData> = input.iter().map(|c| OutputData{z_value: -0.0, layer_output: c.clone()}).collect();
        // Write input layer to first row of output buffer
        // compute.write_buffer(&state.queue, &input, 0, 0);
        compute.write_buffer_raw(&state.queue, &cooked_input, 0, 0);

        for i in 0..self.layers.len() {

            let flat_weights: Vec<f32> = self.layers[i].weights.clone().into_iter().flatten().collect();
            // Write the new weights matrix
            compute.write_buffer(&state.queue, &bytemuck::cast_slice(flat_weights.as_slice()).to_vec(), 0, 2);
            // Write BufferData
            let buffer_data = BufferData::new_from_layer(&input, &self.layers, (i+1).try_into().unwrap());
            let buffer_data_bytes: Vec<u8> =  buffer_data.to_bytes();
            // Write to BufferData to buffer_data_buffer
            compute.write_buffer_bytes(&state.queue, &buffer_data_bytes, 0, 3);
            // Multiply the matrix and put output into output_buffer[i+1];
            // Then run activation function and replace output[i+1] with result;
            // TODO ADD ACTIVATION FUNCTION TO SHADER + USE ENUM TO SEND LAYERS' activation functions to gpu
            compute.calculate(&state.device, &state.queue);
             
            
        }

        // FINISH: FIXING TO OUTPUT Z VALUES AND VALUES SEPARATELY | DONE
        // FINISH: ENSURE THAT INPUT IS PUT INTO THE layer_output rather than z_value | DONE     
        // FINISH: UPDATE LAYER INPUTS AND OUTPUTS 
        // let mut flat_output: Vec<OutputData> = compute.read_output(&state.queue, &state.device).await
        //     .as_slice().chunks(2)
        //     .map(|c| OutputData{z_value: c[0], layer_output: c[1]})
        //     .collect();
        let mut flat_output: Vec<OutputData> = bytemuck::cast_slice(compute.read_output(&state.queue, &state.device).await.as_slice()).to_vec();

        let mut output: Vec<Vec<OutputData>> = Vec::new();
        // let chunk = input.clone();
        let chunk = cooked_input;
        output.push(chunk);
        // for l_size in [input.iter(), self._layers.iter().map(|i| i.size)].iter().flatten() {
        for l_size in layer_sizes.into_iter() {
            let chunk: Vec<OutputData> = bytemuck::cast_slice(flat_output.drain(0..(l_size) as usize).as_slice()).to_vec();
            output.push(chunk);
        }
        

        // log::info!("Compute: {:?}", test);
        // event_loop.run(move |_event, _control_flow| {}).unwrap();

        // let result: Vec<Vec<f32>> = Vec::new();

        // return result;
        return output.last().unwrap().into_iter().map(|c| c.layer_output).collect();
    }

    async fn backpropagate(&mut self, actual_classification: &Vec<f32>) -> f32 {
        todo!();
    }
    // async fn backpropagate(&mut self, actual_classification: &Vec<f32>) -> impl Future<Output = f32> {
        
    // }

    async fn train(epochs : usize) {
        
    }
}

pub trait Network 
where Self: Sized {

    fn get_layers(self:&Self) -> &Vec<Layer>;
    fn get_layers_mut(self:&mut Self) -> &mut Vec<Layer>;

    // pub layers: Vec<Layer>;
    /// Generates network and auto fills layers from some kind of file storing weights and biases
    /// 
    /// Perhaps we use a file organized like this:
    /// 
    /// Header with magic number, how many layers, nodes per layer
    /// 
    /// With use a file with 8 * 28 bytes per weight row, with 28 rows (assuming layers the same size as our input layer).
    /// 
    /// Finally have an optional section with a delimeter and then N rows of the biases, with N being the number of layers.
    /// 
    fn new_from_file(_file_name: &str)-> std::io::Result<Self> {
        todo!();
    }

    fn write_to_file(file_name: &str) -> std::io::Result<()> {
        todo!()
    }

    fn new_default() -> Self;

    fn new(learning_rate: f32) -> Self;
    
    // pub fn first_layer(self: &Self) -> Option<&Layer> {
    //     self.layers.first()
    // }
    
    fn output_layer(self: &Self) -> Option<&Layer> {
        self.get_layers().last()
    }

    fn add_layer(self: &mut Self, layer: Layer) -> Result<(), String> {
        match self.get_layers().last() {
            // This code might be wrong lol
            Some(prev_layer) if prev_layer.size != layer.weights[0].len() => {
                return Err(std::format!(
                    "Rows in previous layer (size) does not match the number of columns in weights of new layer! Previous layer size: {}  |  New weight matrix dims: ({}, {})", 
                    prev_layer.size, layer.weights.len(), layer.weights[0].len()
                ));
            }    
            None => {}
            _ => {}
        }

        self.get_layers_mut().push(layer);

        Ok(())    
    }
    fn set_hidden_activation(self: &mut Self, func: &ActivationFunction) {
        for i in 0..self.get_layers().len()-1 {
            let layer: &mut Layer = self.get_layers_mut().get_mut(i).expect("No layers in network!");
            layer.activation_function = func.clone();
        }
    }

    fn set_output_activation(self: &mut Self, func: &ActivationFunction) {
        self.get_layers_mut().last_mut().expect("No layers in network!").activation_function = func.clone();
    }

    /// Each vector in output represents one individual layer's output
    /// Must return just the output layer's output
    /// Must also ensure that layer inputs, z-values, and outputs are correctly updated, though isn't nesisarrily required to run layer.calculate()
    async fn calculate(self: &mut Self, input: &Vec<f32>) -> Vec<f32>;


    async fn backpropagate(&mut self, actual_classification: &Vec<f32>) -> f32;
    async fn train(epochs : usize) {
        todo!();
    }
}

#[derive(Clone)]
pub struct ClassicNetwork/*<const dim: NetworkDim = NetworkDim::Invalid>*/ /*<I: Iterator<Item=[[f64; 28]; 28]>>*/ {
    // _num_layers: usize,
    // _input: [[f64;28]; 28],
    // _output: [[f64;28];28],

    // _input_layer: Option<&'a Layer>,
    input: Vec<f32>,
    output: Vec<f32>,
    layers: Vec<Layer>,
    learning_rate: f32,
    // _output_layer: Option<&'a Layer>,


    // vec<[[f64;]]>
}

impl Network for ClassicNetwork {

    fn new(learning_rate: f32) -> Self {
        let network = Self {
            layers : vec![],
            learning_rate,
            input : vec![],
            output : vec![],
        };
        return network;
    }

    fn new_default() -> Self {
        let mut network = ClassicNetwork {
            layers : vec![],
            learning_rate : 0.1,
            input : vec![],
            output : vec![],
        };
        // single layer neural network
        let first_layer = Layer::new_random
        (784, 400, ActivationFunction::Relu);
        let output_layer = Layer::new_random
        (400, 10, ActivationFunction::Softmax);
        network.layers.push(first_layer);
        network.layers.push(output_layer);
        return network;
    }

    fn get_layers(self: &Self) -> &Vec<Layer> {
        return &self.layers;
    }
    fn get_layers_mut(self: &mut Self) -> &mut Vec<Layer> {
        return &mut self.layers;
    }
     
    // fn new() -> Self {
    //     todo!();
    //     // Self{
    //     //     layers : vec![],
    //     // }
    // }

    async fn calculate(self: &mut Self, input: &Vec<f32>) -> Vec<f32> {
        self.input = input.clone();
        let mut curr_input: Vec<f32> = input.clone();

        assert!(self.output_layer().expect("No layers!").size == 10, "output layer has incorrect size! Expected {}, found {}", 10, self.output_layer().expect("No layers!").size);
        
        // let curr_input_matrix:Array2<f64> = Array2::from_shape_vec((1,input.len()), input).expect("Input array bad!");
        
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len()+1);

        result.push(curr_input.clone());

        for curr_layer in self.layers.iter_mut() {
            curr_input = curr_layer.calculate(&curr_input);
        }
        self.output = curr_input.clone();
        return curr_input;
    } 
    
    /// actual_classification is the vector of 10 values containing 0s 
    /// and 1 for the actual number the model was supposed to predict
    /// CLARIFY: What is the meaning of the output here?
    async fn backpropagate(&mut self, actual_classification: &Vec<f32>) -> f32 {
        let fake_zero = 1e-10;
        let loss: f32 = self
            .output
            .iter()
            .zip(actual_classification)
            .map(|(&output, &target)| -(target * (output.max(fake_zero).ln())))
            .sum::<f32>();        
        // Compute the gradient of the loss (cross-entrophy) bc we are using softmax
        // gradient is with respect to pre-softmax values
        // literally just sum of actual - predicted
        let output_gradient: Vec<f32> = self
            .output
            .iter()
            .zip(actual_classification)
            .map(|(&output, &target)| output - target) // Gradient for softmax with cross-entropy
            .collect();

        // Perform backpropagation for each layer
        let mut gradient = output_gradient; // Initialize with the gradient of the output layer
        for i in (0..self.layers.len()).rev() {
            gradient = self.layers[i].backpropagate(gradient, self.learning_rate);
        }
        return loss;
    }

    async fn train(epochs : usize) {
        todo!();
    }
}

/// More positive weights are more green, and more negitive weights are more red. 
/// 
/// Weights close to 0 are more black.
fn weights_to_ppm(_file_name: &str) {
    todo!();
}


/// Each image is 28x28 pixels, with values from 0 to 255
/// 
/// Each row of the file is its own digit image with 748 bytes
/// 
/// Should return a 28x28 array of values corresponding to the image on the given row of the file
fn ubyte_file_read(_input_file: &str, _row:usize) -> [[f32;IMG_WIDTH]; IMG_HEIGHT] {
    todo!();
}


/// Scores output of function. Ideally a more acurate 
pub trait ScoreFunction {
    fn calculate(&self) -> f32;
}

// pub fn ScoreFunction() {
//     todo!();
// }

/// Accepts an array reference. Each element of the array is the relative probability of that digit digit being selected.
/// 
/// The output's indexes match to their corresponding digit, so a bigger value for 0, means a bigger chance of the digit being tested being a zero.
/// 
/// After normalization, each value should be between 0 and 1, such that the values maintain their ratio and the sum of all the values sum to 1
/// 
/// This ideally is done by dividing each value by the magnitude of the 10th dimensional array
/// 
/// If all input values are 0, returns all 0.0
///
/// Returns the normalized array. 
pub fn normalize_output(output: &Vec<f32>) -> Vec<f32> {
// fn normalize_output(output: &[f64; 10]) -> [f64;10] {
    let size: usize = output.len();
    let sum: f32 = output.iter().sum();
    
    // if sum == 0.0 {return [0.0; 10]}
    if sum == 0.0 {return vec![0.0; size];}
    
    // let mut result: [f64; 10] = [0.0; 10];
    let mut result: Vec<f32> = Vec::with_capacity(size);
    for val in output.iter() {
        result.push(val/sum);
        // result[i] = val/sum;
    }
    result
}

// fn([[f64;28]; 28]);

fn compute_loss(predicted : Vec<f64>, actual : Vec<f64>) -> f64 {
    todo!();
}