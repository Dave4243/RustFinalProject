use std::f64::consts::E;
use std::vec::Vec;
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
    pub fn calculate(self, input: f64) -> f64 {
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
    weights: Vec<Vec<f64>>,

    /// The bias for each neuron. Should be added to the dot product of 
    /// previous weights and current weights
    biases: Vec<f64>,

    activation_function: ActivationFunction,
}

impl Layer { 
    // initalizes random weights and biases, 
    // input size is the size of the previous layer
    // output size is the size of this layer (how many neurons in the layer)
    fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        
        // Randomly initialize weights and biases
        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();

        let biases: Vec<f64> = (0..output_size).map(|_| rng.gen_range(-0.5..0.5)).collect();

        Self {
            size : input_size,
            weights,
            biases,
            activation_function : activation,
        }
    }

    // function to initiate a forward pass through this layer of neurons
    pub fn calculate(self: &Self, prev_layer_out: &Vec<f64>) -> Vec<f64> {
        // Input vector is vertical
        let mut result: Vec<f64> = vec![0.0; self.size]; 

        // Col is the index of the layer vector
        for (col, weights_row) in self.weights.iter().enumerate() {
            let mut sum: f64 = weights_row.iter()
                .zip(prev_layer_out)
                .map(|(x, y)| x*y)
                .sum();
            sum += self.biases[col];
            result[col] = sum;
        }
        
        // we see a problem arise if "x" is too large or too small (saturated network)
        // because the sigmoid will map it to -1 or 1, and the gradient
        // may vanish as a result. There are techniques to combat this issue
        // but we've initalized weights and biases to [-0.5, 0.5] for now.
        match self.activation_function {
            ActivationFunction::Softmax => {
                // Find the maximum value for numerical stability
                let max = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = result.iter().map(|&x| (x - max).exp()).collect::<Vec<f64>>();
                let sum: f64 = exps.iter().sum();
                exps.iter().map(|&x| x / sum).collect::<Vec<f64>>();
            },
            _ => result.iter_mut().for_each(|x| *x = self.activation_function.calculate(*x)),
        }
        return result;
    }

    /// Backprop works as follows:
    /// Take the output_gradient (gradient of loss w/ respect to current neuron output)
    /// Multiply output_gradient by derivative of activation function "a".
    /// This gives you dL/dz (chain rule: dL/dOutput * dOutput/dz (a')= dL/dz)
    ///     dL/dz is how much the loss changes given a change in z
    /// Now that you have dL/dz, you can get dL/dW = dL/dz * dz/dW.
    ///     dL/dW is how much the error changes given a change in w
    /// This is where input of the function from forward pass comes in handy, let's say input = "x"
    /// z = W*x + b, so dz/dW = x
    /// Thus, dL/dz * x = dL/dW, and then you can adjust the weight accordingly (backprop)
    ///     dL/dW = dL/dz * x
    /// For the bias, dz/db is just 1, so dL/db = dL/dz * dz/db = dL/dz :)
    ///     dL/db = dL/dz
    /// 
    /// Now that we are done with backprop, we need to pass dL/dPrevLayerOutput to the prev layer
    /// recall PrevLayerOutput = "x" from earlier. We need dL/dx.
    /// Again chain rule gives dL/dx = dL/dz * dz/dx.
    /// z = W * x + b, so dz/dx = W. So dL/dx = dL/dz * W
    ///     dL/dx = dL/dz * W
    /// Do this for every neuron
    fn backpropagate(&mut self, input: &Vec<f64>, output_gradient: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        todo!();
    }
}

#[derive(Clone)]
pub struct Network {
    layers: Vec<Layer>,
    learning_rate : f64,
}

impl Network{
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
        let mut network = Network {
            layers : vec![],
            learning_rate : 0.1,
        };
        // single layer neural network
        let first_layer = Layer::new
        (784, 400, ActivationFunction::Relu);
        let output_layer = Layer::new
        (400, 10, ActivationFunction::Softmax);
        network.layers.push(first_layer);
        network.layers.push(output_layer);
        return network;
    }

    pub fn first_layer(self: &Self) -> Option<&Layer> {
        self.layers.first()
    }
    
    pub fn output_layer(self: &Self) -> Option<&Layer> {
        self.layers.last()
    }

    pub fn add_layer(self: &mut Self, layer: Layer) -> Result<(), String> {
        match self.layers.last() {
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

        self.layers.push(layer);

        Ok(())    
    }

    pub fn set_hidden_activation(self: &mut Self, func: &ActivationFunction) {
        for i in 0..self.layers.len()-1 {
            let layer: &mut Layer = self.layers.get_mut(i).expect("No layers in network!");
            layer.activation_function = func.clone();
        }
    }
    
    pub fn set_output_activation(self: &mut Self, func: &ActivationFunction) {
        self.layers.last_mut().expect("No layers in network!").activation_function = func.clone();
    }

    // changed function to return a 2d vector of the values during forward apss
    pub fn calculate(self: &Self, input: &Vec<f64>) -> Vec<Vec<f64>>/*[f64; 10]*/ {
        let mut curr_input_matrix: Vec<Vec<f64>> = vec![input.clone()];

        for curr_layer in self.layers.iter() {
            let next_input = curr_layer.calculate(curr_input_matrix.last().unwrap());
            curr_input_matrix.push(next_input);
        }
        return curr_input_matrix;
    } 

    pub fn backpropagate(self : &Self, activations : Vec<Vec<f64>>, actual : Vec<f64>) {
        todo!();
    }

    pub fn train(epochs : usize) {
        todo!();
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
fn normalize_output(output: &Vec<f64>) -> Vec<f64> {
// fn normalize_output(output: &[f64; 10]) -> [f64;10] {
    let size: usize = output.len();
    let sum: f64 = output.iter().sum();
    
    // if sum == 0.0 {return [0.0; 10]}
    if sum == 0.0 {return vec![0.0; size];}
    
    // let mut result: [f64; 10] = [0.0; 10];
    let mut result: Vec<f64> = Vec::with_capacity(size);
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