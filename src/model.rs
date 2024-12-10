
use std::{error::Error, io, process};
use na::{U2, U3, Dynamic, ArrayStorage, VecStorage, Matrix};
use std::f64::consts::E;
use std::vec::Vec;
use std::fs::{File, OpenOptions};
use std::io::{Cursor, BufRead, Read, Write};
use byteorder::{BigEndian, ReadBytesExt};
use rand::Rng;

// std::f64::consts::E;

pub struct UbyteFile {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl UbyteFile {
    fn new(f: &str) -> Result<UbyteFile, Box<dyn Error>> {
        let mut file = File::open(f)?;
        let mut file_str: Vec<u8> = Vec::new();
        file.read_to_end(&mut file_str)?;
        let mut r = Cursor::new(file_str);

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        let magic_number = r.read_i32::<BigEndian>()?;
        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                if sizes[1] != 28 || sizes[2] != 28 {
                    panic!("Images are not 28x28 pixels");
                }
            }
            _ => panic!("Magic number is not valid"),
        }
        r.read_to_end(&mut data)?;
        Ok(UbyteFile{sizes, data})
    }

    fn ubyte_file_read(input_file: &str, row:usize) -> [[f64;28]; 28] {
        let input = UbyteFile::new(input_file).unwrap();
        let image_shape = (input.sizes[1] * input.sizes[2]) as usize;
        let index = image_shape*row as usize;
        let mut read_file_vec: [[f64;28]; 28] = [[0.0;28]; 28];
        for i in 0..28 {
            for j in 0..28 {
                let idx = index + i*28 + j;
                let data: f64 = input.data[idx] as f64 / 255.0;
                read_file_vec[i][j] = data;
            }
        }
        read_file_vec
    }

    fn get_image_vec(&self, input_file: &str) -> Vec<[[f64;28]; 28]> {
        let mut image_vec: Vec<[[f64;28]; 28]> = Vec::new();
        for i in 0..self.sizes[0] {
            image_vec.push(ubyte_file_read(input_file, i as usize));
        }
        image_vec
    }

    fn get_labels_vec(&self, input_file: &str) -> Vec<u8> {
        let labels = UbyteFile::new(input_file);
        labels.unwrap().data
    }
}

// pub mod model {
// }
static B:f64 = 1.0;

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

// trait ActivationFunction {
//     fn calculate(input:f64, min:f64, max:f64) -> f64;
// }

// pub struct Sigmoid {}

// impl ActivationFunction for Sigmoid {
//     fn calculate(input:f64, min:f64, max:f64) -> f64 {
//         1.0/(E.powf(-input * B))
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
    size: usize,

    weights: Vec<Vec<f64>>,

    biases: Vec<f64>,

    input : Vec<f64>,

    z_values : Vec<f64>,

    output : Vec<f64>,

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
            input : vec![],
            z_values : vec![],
            output: vec![],
            activation_function : activation,
        }
    }

    // function to initiate a forward pass through this layer of neurons
    pub fn calculate(&mut self, prev_layer_out: &Vec<f64>) -> Vec<f64> {
        self.input = prev_layer_out.clone();
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
        // Store the pre-activation values
        self.z_values = result.clone();
        
        // we see a problem arise if "x" is too large or too small (saturated network)
        // because the sigmoid will map it to -1 or 1, and the gradient
        // may vanish as a result. There are techniques to combat this issue
        // but we've initalized weights and biases to [-0.5, 0.5] for now.
        match self.activation_function {
            ActivationFunction::Softmax => {
                // Find the maximum value for numerical stability
                let max = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = result.iter().map(|&x| (x - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                result = exps.iter().map(|&x| x / sum).collect();
            },
            _ => result.iter_mut().for_each(|x| *x = self.activation_function.calculate(*x)),
        }
        // Store results and return
        self.output = result.clone();
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
    fn backpropagate(&mut self, output_gradient: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let dLdz : Vec<f64>;
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

#[derive(Clone)]
pub struct Network {
    layers: Vec<Layer>,
    learning_rate : f64,
    input : Vec<f64>,
    output : Vec<f64>,
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
    pub fn new_from_file(&mut self, file_name: &str) -> std::io::Result<Self> {
        let file = File::open(file_name)?;
        let reader = io::BufReader::new(file);

        let mut layers = Vec::new();
        let mut lines = reader.lines();
        
        let magic_number = lines.next().unwrap()?.trim().parse::<i32>().expect("Failed to parse magic number");
        // reader.read_line(&mut magic_number_str);
        // let magic_number: i32 = magic_number_str.trim().parse::<i32>().expect("Failed to parse magic number");
        if magic_number != 3021 {
            panic!("Magic number is invalid");
        }

        let learning_rate: f64 = lines.next().unwrap()?.trim().parse::<f64>().expect("Failed to parse learning rate");
        let num_layers: usize = lines.next().unwrap()?.trim().parse::<usize>().expect("Failed to parse number of nodes");
        
        // let mut layer_size: usize = 0;

        for _ in 0..num_layers {
            //let layer_size: usize = lines.next().unwrap()?.parse().unwrap();
            let layer_size = lines.next().unwrap()?;
            let size: usize = layer_size.split_whitespace().last().unwrap().parse::<usize>().unwrap();

            let activation_function = match lines.next().unwrap()?.as_str() {
                "Activation function: Sigmoid" => ActivationFunction::Sigmoid,
                "Activation function: Tanh" => ActivationFunction::Tanh,
                "Activation function: Relu" => ActivationFunction::Relu,
                "Activation function: Softmax" => ActivationFunction::Softmax,
                "Activation function: None" => ActivationFunction::None,
                _ => panic!("Unknown activation function"),
            };

            let mut weights = Vec::new();
            loop {
                let weights_label = lines.next().unwrap()?;
                if weights_label.trim().is_empty() {
                    break;
                }
                let weights_str = lines.next().unwrap()?;
                let weights_vec: Vec<f64> = weights_str.split(',').map(|x| x.trim().parse::<f64>().unwrap()).collect();
                weights.push(weights_vec);
            }

            let mut biases: Vec<f64> = lines.next().unwrap()?.split(',').map(|x| x.trim().parse::<f64>().unwrap()).collect();
            let mut input: Vec<f64> = lines.next().unwrap()?.split(',').map(|x| x.trim().parse::<f64>().unwrap()).collect();
            let mut z_values: Vec<f64> = lines.next().unwrap()?.split(',').map(|x| x.trim().parse::<f64>().unwrap()).collect();
            let mut output: Vec<f64> = lines.next().unwrap()?.split(',').map(|x| x.trim().parse::<f64>().unwrap()).collect();
            // loop {
            //     let biases_label = lines.next().unwrap();
            //     if biases_label?.trim().is_empty() {
            //         break;
            //     }
            //     let bias_str = lines.next().unwrap();
            //     let bias: f64 = bias_str?.trim().parse::<f64>().unwrap();
            //     biases.push(bias);
            // }
            layers.push(Layer { 
                size: size,
                weights: weights,
                biases: biases,
                input: input,
                z_values: z_values,
                output: output,
                activation_function: activation_function,
            })
        }

        let network = Self {
            layers: layers,
            learning_rate: learning_rate,
            input: vec![],
            output: vec![],
        };
        Ok(network)
        // Ok(Network{_input_layer = layers, learinng_rate = learning_rate})
        // for line in lines {
        //     if line.trim().is_empty(){
        //         layers.push(Layer {
        //             size: layer_size,
        //             weights: current_layer_weights.clone(),
        //             biases: current_layer_biases.clone(),
        //         });
        //         layer_size = 0;
        //         current_layer_weights.clear();
        //         current_layer_biases.clear();
        //     }
        // }
    }

    pub fn write_to_file(&self, file_name: &str) -> std::io::Result<()> {
        let mut file = File::create(file_name)?;
        // let file = OpenOptions::new().append(true).open(file_name).expect("Unable to open file");
        // file.write_all(b"{}", 3021 as i32)?;
        writeln!(file, "{}", 3021 as i32);
        writeln!(file, "{}", self.learning_rate as f64);
        writeln!(file, "{}", self.layers.len() as usize);
        for layer in &self.layers {
            writeln!(file, "Layer size: {}", layer.size);
            match &layer.activation_function {
                ActivationFunction::Sigmoid => writeln!(file, "Activation function: Sigmoid")?,
                ActivationFunction::Tanh => writeln!(file, "Activation function: Tanh")?,
                ActivationFunction::Relu => writeln!(file, "Activation function: Relu")?,
                ActivationFunction::Softmax => writeln!(file, "Activation function: Softmax")?,
                ActivationFunction::None => writeln!(file, "Activation function: None")?,
            }
            for weight in &layer.weights {
                writeln!(file, "Weights");
                let weight_str = weight.iter().map(|x| format!("{:.9}", x)).collect::<Vec<String>>().join(", ");
                writeln!(file, "[{}]", weight_str)?;
            }

            writeln!(file, "Bias");
            let bias_str = layer.biases.iter().map(|x| format!("{:.9}", x)).collect::<Vec<String>>().join(",");
            writeln!(file, "[{}]", bias_str)?;

            // for bias in &layer._biases {
            //     writeln!(file, "Bias");
            //     writeln!(file, "{:.4}", bias)?;
            // }

            writeln!(file, "Input layer");
            let input_str = layer.input.iter().map(|x| format!("{:.9}", x)).collect::<Vec<String>>().join(",");
            writeln!(file, "[{}]", input_str);

            writeln!(file, "Z-values");
            let z_str = layer.z_values.iter().map(|x| format!("{:.9}", x)).collect::<Vec<String>>().join(",");
            writeln!(file, "[{}]", z_str);

            writeln!(file, "Output");
            let output_str = layer.output.iter().map(|x| format!("{:.9}", x)).collect::<Vec<String>>().join(",");
            writeln!(file, "[{}]", output_str);

            writeln!(file, "")?;
        }
        
        Ok(())
    }

    pub fn new() -> Self {
        let mut network = Network {
            layers : vec![],
            learning_rate : 0.1,
            input : vec![],
            output : vec![],
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

    pub fn calculate(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.input = input.clone();
        let mut curr_input: Vec<f64> = input.clone();

        for curr_layer in self.layers.iter_mut() {
            curr_input = curr_layer.calculate(&curr_input);
        }
        self.output = curr_input.clone();
        return curr_input;
    } 
    
    // actual_classification is the vector of 10 values containing 0s 
    // and 1 for the actual number the model was supposed to predict
    pub fn backpropagate(&mut self, actual_classification: &Vec<f64>) -> f64 {
        let fake_zero = 1e-10;
        let loss: f64 = self
            .output
            .iter()
            .zip(actual_classification)
            .map(|(&output, &target)| -(target * (output.max(fake_zero).ln())))
            .sum::<f64>();        
        // Compute the gradient of the loss (cross-entrophy) bc we are using softmax
        // gradient is with respect to pre-softmax values
        // literally just sum of actual - predicted
        let output_gradient: Vec<f64> = self
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
// fn ubyte_file_read(input_file: &str, row:usize) -> [[f64;28]; 28] {
//     let data = &UbyteFile::new(&str)
// }

fn ubyte_file_read(input_file: &str, row:usize) -> [[f64;28]; 28] {
    let input = UbyteFile::new(input_file).unwrap();
    let image_shape = (input.sizes[1] * input.sizes[2]) as usize;
    let index = image_shape*row as usize;
    let mut read_file_vec: [[f64;28]; 28] = [[0.0;28]; 28];
    for i in 0..28 {
        for j in 0..28 {
            let idx = index + i*28 + j;
            let data: f64 = input.data[idx] as f64 / 255.0;
            read_file_vec[i][j] = data;
        }
    }
    read_file_vec
}

fn label_read (input_file: &str) -> Vec<u8> {
    let input = UbyteFile::new(input_file).unwrap();
    input.data
}

/// Scores output of function. Ideally a more acurate 
pub fn ScoreFunction() {
    todo!();
}

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

