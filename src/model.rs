// use std::{error::Error, io, process};
// use na::{U2, U3, Dynamic, ArrayStorage, VecStorage, Matrix};
// use ndarray::Array2;
#![allow(dead_code)]

use itertools::{interleave, Itertools};
use std::{f32::consts::E, num::NonZero};
use std::vec::Vec;

use rust_webgpu::{start_logger, Compute, State};


const IMG_WIDTH: usize = 28;
const IMG_HEIGHT: usize = 28;


// std::f64::consts::E;

// pub mod model {
// }
#[derive(Copy, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    None,
}
impl ActivationFunction {
    pub fn calculate(self, input: f32) -> f32 {
        match self {
            Self::Sigmoid => {
                const B:f32 = 1.0;
                1.0/(E.powf(-input * B))
            },
            Self::Tanh => {
                (E.powf(input)-E.powf(-input)) / (E.powf(input) + E.powf(-input))
            },
            Self::Relu => todo!(),
            Self::None => input,
        }
    }
}

// pub trait ActivationFunction {
//     fn calculate(input:f64) -> f64;
// }

// pub struct Sigmoid {}
// impl ActivationFunction for Sigmoid {
//     fn calculate(input:f64) -> f64 {
//         const B:f64 = 1.0;
        
//         1.0/(E.powf(-input * B))
//     }
// }

// pub struct Tanh {}
// impl ActivationFunction for Tanh {
//     fn calculate(input:f64) -> f64 {
//        (E.powf(input)-E.powf(-input)) / (E.powf(input) + E.powf(-input))
//     }
// }

/// Describes 1D vector layer. 
/// 
/// Each layer contains the weights that relate it to its previous layer.
/// 
/// The matrix multiplation of the previous layer's output and this layer's weights matrix results in this layers output 
/// 
/// Also contains optional biases, there is a bias for each node in the layer. This repesents a flat amount to add to the activation sum after accounting for weights
#[derive(Clone)]
pub struct Layer {
    // _nodes: [f64; layer_size],

    pub size: usize,
    // _prev_layer_size: Option<usize>,

    weights: Vec<Vec<f32>>, // layer_size,
    // _weights: Array2<f64>, // layer_size,

    /// The bias for each node. Should be added to the sum of 
    pub biases: Option<Vec<f32>>,
    // _biases: Option<Array2<f64>>,

    activation_function: ActivationFunction,
    // _data: [f64; N] = [0.0; N];
    // _weights: [f64; N] = [0.0; N];

    // Layer()

    // Layer 

}

// Add method for merging two layers, for use in convolutions
impl Layer { 
    pub fn new(
        size: usize,
        weights: Vec<Vec<f32>>,
    ) -> Self {
        // self._activation_function = ActivationFunction::None;
        // self._biases = None;
        assert!(weights.len() == size, "Number of rows (length of columns) in weight, {}, does not match output layer size, {}!", weights.len(), size);
        
        Self{
            size: size,
            weights,
            biases : None,
            activation_function : ActivationFunction::None,
        }
    }

    pub fn calculate(self: &Self, prev_layer_out: Vec<f32>) -> Vec<f32> {
        assert!(self.weights.len() == self.size, "Number of rows (length of columns) in weight, {}, does not match output layer size, {}!", self.weights.len(), self.size);
        
        // Input vector is vertical
        let mut result: Vec<f32> = vec![0.0; self.size]; 

        // Col is the index of the layer vector
        for (col, weights_row) in self.weights.iter().enumerate() {
            let mut sum: f32 = weights_row.iter().zip(&prev_layer_out).map(|(x, y)| x*y).sum();
            if let Some(bias) = &self.biases {sum += bias[col]}
            result[col] = sum;
        }
        return result;
    }
}


enum NetworkDim {
    Invalid = -1,
    Network2D = 2,
    Network3D = 3,
}



// #[derive(Clone)]
pub struct ComputeNetwork<'a> {
    _layers: Vec<Layer>,
    state: Option<State<'a>>,
    compute: Option<Compute<f32>>,
}

// struct BufferData {
//     prev_layer_size: u32,
//     curr_layer_size: u32,
//     weights_dims: [u32; 2],
//     curr_layer_index: u32,
//     // output_data_sizes: array<u32>
// }

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

impl Network for ComputeNetwork<'static> {

    fn get_layers(self: &Self) -> &Vec<Layer> {
        return &self._layers;
    }
    fn get_layers_mut(self: &mut Self) -> &mut Vec<Layer> {
        return &mut self._layers;
    }

    fn new() -> Self { 
        Self{
            _layers : vec![],
            // State and compute are lazy initialized, they cannot be sized until _layers is complete at caluclation time
            state: None,
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
    /// 
    async fn calculate(self: &mut Self, input: Vec<f32>) -> Vec<Vec<f32>>{
        start_logger();
    
        // let event_loop = EventLoop::new().unwrap();

        #[allow(unused_mut)]
        if self.state.is_none() {
            self.state = Some(State::new_no_graphics().await);
        }
        let state: &State = self.state.as_ref().unwrap();

        let mut max_layer_size = input.len();
        for layer in self._layers.iter() {
            if layer.size > max_layer_size {max_layer_size = layer.size;}
        }

        let mut max_weight_size = 0;
        for layer in self._layers.iter() {
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
        self._layers.iter().for_each(|i| layer_sizes.push(i.size as u32));
        // This has to be the sum of the sizes of every layer, including the input, since that one isnt caluclated by the shader.
        // let output_size = NonZero::new(self._layers.iter().map(|i| i.size as u32).sum()).unwrap();
        let output_size: u32 = layer_sizes.iter().sum();

        let shader_str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/matrix.wgsl"));        
        let mut compute: Compute<f32> = Compute::new(&state.device, shader_str, input_sizes, output_size.try_into().unwrap());

        // Set to calculate layer 1 (layer 1 being the one after the input layer, this is the same as _layers[0]) 
        // let buffer_data = BufferData::new_from_layer(&input, &self._layers, 1);
        // let buffer_data_bytes: Vec<u8> =  buffer_data.to_bytes();

        // Write input layer to first row of output buffer
        compute.write_buffer(&state.queue, &input, 0, 0);

        for i in 0..self._layers.len() {

            let flat_weights: Vec<f32> = self._layers[i].weights.clone().into_iter().flatten().collect();
            // Write the new weights matrix
            compute.write_buffer(&state.queue, &bytemuck::cast_slice(flat_weights.as_slice()).to_vec(), 0, 2);
            // Write BufferData
            let buffer_data = BufferData::new_from_layer(&input, &self._layers, (i+1).try_into().unwrap());
            let buffer_data_bytes: Vec<u8> =  buffer_data.to_bytes();
            // Write to BufferData to buffer_data_buffer
            compute.write_buffer_bytes(&state.queue, &buffer_data_bytes, 0, 3);
            // Multiply the matrix and put output into output_buffer[i+1];
            // Then run activation function and replace output[i+1] with result;
            // TODO ADD ACTIVATION FUNCTION TO SHADER + USE ENUM TO SEND LAYERS' activation functions to gpu
            compute.calculate(&state.device, &state.queue);
             
            
        }

        
        let mut flat_output = compute.read_output(&state.queue, &state.device).await;

        let mut output: Vec<Vec<f32>> = Vec::new();
        let chunk = input.clone();
        output.push(chunk);
        // for l_size in [input.iter(), self._layers.iter().map(|i| i.size)].iter().flatten() {
        for l_size in layer_sizes.into_iter() {
            let chunk: Vec<f32> = bytemuck::cast_slice(flat_output.drain(0..l_size as usize).as_slice()).to_vec();
            output.push(chunk);
        }
        

        // log::info!("Compute: {:?}", test);
        // event_loop.run(move |_event, _control_flow| {}).unwrap();

        // let result: Vec<Vec<f32>> = Vec::new();

        // return result;
        return output;
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
    fn write_to_file(_file_name: &str)-> std::io::Result<()> {
        todo!();
    }
    fn new() -> Self;

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
    async fn calculate(self: &mut Self, input: Vec<f32>) -> Vec<Vec<f32>>;
}

#[derive(Clone)]
pub struct ClassicNetwork/*<const dim: NetworkDim = NetworkDim::Invalid>*/ /*<I: Iterator<Item=[[f64; 28]; 28]>>*/ {
    // _num_layers: usize,
    // _input: [[f64;28]; 28],
    // _output: [[f64;28];28],

    // _input_layer: Option<&'a Layer>,
    _layers: Vec<Layer>,
    // _output_layer: Option<&'a Layer>,


    // vec<[[f64;]]>
}

impl Network for ClassicNetwork {

    fn get_layers(self: &Self) -> &Vec<Layer> {
        return &self._layers;
    }
    fn get_layers_mut(self: &mut Self) -> &mut Vec<Layer> {
        return &mut self._layers;
    }
     
    fn new() -> Self {
        Self{
            _layers : vec![],
        }
    }

    async fn calculate(self: &mut Self, input: Vec<f32>) -> Vec<Vec<f32>> {

        assert!(self.output_layer().expect("No layers!").size == 10, "output layer has incorrect size! Expected {}, found {}", 10, self.output_layer().expect("No layers!").size);
        
        // let curr_input_matrix:Array2<f64> = Array2::from_shape_vec((1,input.len()), input).expect("Input array bad!");
        let mut curr_input_matrix: Vec<f32> = input.clone();
        
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(self._layers.len()+1);

        result.push(curr_input_matrix.clone());

        for curr_layer in self._layers.iter() {
            let raw = curr_layer.calculate(curr_input_matrix);
            let cooked = raw.into_iter().map(|x| curr_layer.activation_function.calculate(x)).collect();
            curr_input_matrix = cooked;
            result.push(curr_input_matrix.clone());
        }

        // let result: Vec<f32> = curr_input_matrix;
        assert!(result.last().unwrap().len() == self._layers.last().unwrap().size, "Wrong sized output! Expected size, output layer size: {}, got size {}", self._layers.last().unwrap().size, result.last().unwrap().len());
        
        // normalize_output(&result)

        result

        // return curr_input_matrix.into_raw_vec_and_offset().try_into().expect("Could not convert!")
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

