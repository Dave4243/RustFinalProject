use std::{error::Error, io, process};
// use na::{U2, U3, Dynamic, ArrayStorage, VecStorage, Matrix};
// use ndarray::Array2;
use std::f64::consts::E;
use std::vec::Vec;


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
    pub fn calculate(self, input: f64) -> f64 {
        match self {
            Self::Sigmoid => {
                const B:f64 = 1.0;
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
pub struct Layer {
    // _nodes: [f64; layer_size],

    pub size: usize,
    // _prev_layer_size: Option<usize>,

    weights: Vec<Vec<f64>>, // layer_size,
    // _weights: Array2<f64>, // layer_size,

    /// The bias for each node. Should be added to the sum of 
    biases: Option<Vec<f64>>,
    // _biases: Option<Array2<f64>>,

    activation_function: ActivationFunction,
    // _data: [f64; N] = [0.0; N];
    // _weights: [f64; N] = [0.0; N];

    // Layer()

    // Layer 

}

impl Layer { 
    pub fn new(
        size: usize,
        weights: Vec<Vec<f64>>,
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

    pub fn calculate(self: &Self, prev_layer_out: Vec<f64>) -> Vec<f64> {
        assert!(self.weights.len() == self.size, "Number of rows (length of columns) in weight, {}, does not match output layer size, {}!", self.weights.len(), self.size);
        
        // Input vector is vertical
        let mut result: Vec<f64> = vec![0.0; self.size]; 

        for (col, weights_row) in self.weights.iter().enumerate() {
            let sum: f64 = weights_row.iter().zip(&prev_layer_out).map(|(x, y)| x*y).sum();
            result[col] = sum;
        }
        return result;
    }
}

pub struct Network /*<I: Iterator<Item=[[f64; 28]; 28]>>*/ {
    // _num_layers: usize,
    // _input: [[f64;28]; 28],
    // _output: [[f64;28];28],

    // _input_layer: Option<&'a Layer>,
    _layers: Vec<Layer>,
    // _output_layer: Option<&'a Layer>,


    // vec<[[f64;]]>
}

impl Network {
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
    /// 
    pub fn new_from_file(file_name: &str) -> std::io::Result<Self> {
        todo!()
    }

    pub fn write_to_file(file_name: &str) -> std::io::Result<()> {
        todo!()
    }

    pub fn new() -> Self {
        Network{
            _layers : vec![],
        }
    }

    pub fn input_layer(self: &Self) -> Option<&Layer> {
        self._layers.first()
    }
    
    pub fn output_layer(self: &Self) -> Option<&Layer> {
        self._layers.last()
    }

    pub fn add_layer(self: &mut Self, layer: Layer) -> Result<(), String> {
        match self._layers.last() {
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

        self._layers.push(layer);

        Ok(())    
    }

    pub fn set_hidden_activation(self: &mut Self, func: &ActivationFunction) {
        for i in 0..self._layers.len()-1 {
            let layer: &mut Layer = self._layers.get_mut(i).expect("No layers in network!");
            layer.activation_function = func.clone();
        }
    }
    
    pub fn set_output_activation(self: &mut Self, func: &ActivationFunction) {
        self._layers.last_mut().expect("No layers in network!").activation_function = func.clone();
    }

    pub fn calculate(self: &Self, input: Vec<f64>) -> [f64; 10] {

        assert!(self.output_layer().expect("No layers!").size == 10, "output layer has incorrect size! Expected {}, found {}", 10, self.output_layer().expect("No layers!").size);
        
        // let curr_input_matrix:Array2<f64> = Array2::from_shape_vec((1,input.len()), input).expect("Input array bad!");
        let mut curr_input_matrix: Vec<f64> = input.clone();
        

        for curr_layer in self._layers.iter() {
            let raw = curr_layer.calculate(curr_input_matrix);
            let cooked = raw.into_iter().map(|x| curr_layer.activation_function.calculate(x)).collect();
            // let next_input_matrix: Vec<Vec<f64>> = Vec::with_capacity();
            curr_input_matrix = cooked;
            // let weights_matrix:Array2<f64> = Array2::from(curr_layer._weights);
            // curr_input_matrix = next_input_matrix   
        }

        let result: [f64; 10] = curr_input_matrix.try_into().unwrap_or_else(|x: Vec<f64>| panic!("Wrong sized output! Expected size 10, got size {}", x.len()));
        normalize_output(&result)
        // return curr_input_matrix.into_raw_vec_and_offset().try_into().expect("Could not convert!")
    } 
}

/// More positive weights are more green, and more negitive weights are more red. 
/// 
/// Weights close to 0 are more black.
fn weights_to_ppm(file_name: &str) {
    todo!();
}


/// Each image is 28x28 pixels, with values from 0 to 255
/// 
/// Each row of the file is its own digit image with 748 bytes
/// 
/// Should return a 28x28 array of values corresponding to the image on the given row of the file
fn ubyte_file_read(input_file: &str, row:usize) -> [[f64;IMG_WIDTH]; IMG_HEIGHT] {
    todo!();
}


/// Scores output of function. Ideally a more acurate 
pub trait ScoreFunction {
    fn calculate(&self) -> f64;
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
fn normalize_output(output: &[f64; 10]) -> [f64;10] {
    let sum: f64 = output.iter().sum();
    
    if sum == 0.0 {return [0.0; 10]}
    
    let mut result: [f64; 10] = [0.0; 10];
    for (i, val) in output.iter().enumerate() {
        result[i] = val/sum;
    }
    result
}

// fn([[f64;28]; 28]);

