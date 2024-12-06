use std::f64::consts::E;
use std::vec::Vec;
use rand::Rng;
use std::fs::File;
use std::io::{self, BufReader, Read};

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

    // The bias for each neuron. Should be added to the dot product of 
    // previous weights and current weights
    biases: Vec<f64>,

    // The input from the previous layer ("x")
    input : Vec<f64>,

    // Values z = W*x + b
    z_values : Vec<f64>,

    activation_function: ActivationFunction,
}

impl Layer { 
    // initalizes random weights and biases, 
    // input size is the size of the previous layer
    // output size is the size of this layer (how many neurons in the layer)
    fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        
        // Randomly initialize weights and biases
        // let weights: Vec<Vec<f64>> = (0..output_size)
        //     .map(|_| (0..input_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
        //     .collect();
        let mut rng = rand::thread_rng(); 
        let limit = (2.0 / input_size as f64).sqrt(); 
        let weights: Vec<Vec<f64>> = (0..output_size) .map(|_| (0..input_size).map(|_| rng.gen_range(-limit..limit)).collect()) .collect();

        let biases: Vec<f64> = vec![0.0; output_size];

        Self {
            size : input_size,
            weights,
            biases,
            input : vec![0.0; input_size],
            z_values : vec![],
            activation_function : activation,
        }
    }

    // function to initiate a forward pass through this layer of neurons
    pub fn calculate(&mut self, prev_layer_out: &Vec<f64>) -> Vec<f64> {
        // println!("Layer size: Num neurons (Output){}, Num weights{}", self.weights.len(), self.weights[0].len());
        // println!("Previous layer output: {}", prev_layer_out.len());
        self.input = prev_layer_out.clone();
        // Input vector is vertical
        let mut result: Vec<f64> = vec![0.0; self.weights.len()]; 

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
        // println!("Z values: {:?}", self.z_values);
        
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
        println!("DLDZ: {:?}", dLdz);
        // computes gradient with respect to input (output of prev layer)
        let mut dLdx = vec![0.0; self.input.len()];
        for j in 0..self.input.len() {
            for i in 0..self.weights.len() {
                dLdx[j] += dLdz[i] * self.weights[i][j];
                println!("DLDX FOR OUTPUT NEURON {}: {}", j, dLdx[j]);
            }
        }
        
        // update biases, dL/db = dL/dz * dz/dB = dL/dz * 1 = dL/dz
        for i in 0..self.biases.len() {
            let mut dLdb = dLdz[i];
            println!("DLDB for neuron {}: {} ", i, dLdb);
            self.biases[i] -= learning_rate * dLdb;
        }
        // update weights
        // dL/dW = dL/dz * dz/dW = dL/dz * input
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                let mut dLdW = dLdz[i] * self.input[j];
                println!("DLDW for neuron {}, weight {}: {}", i, j, dLdW);
                self.weights[i][j] -= learning_rate * dLdW;
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
}

impl Network{
    pub fn new() -> Self {
        let mut network = Network {
            layers : vec![],
            learning_rate : 0.01,
        };
        // single layer neural network
        // let first_layer = Layer::new
        // (784, 512, ActivationFunction::Relu);
        // let output_layer = Layer::new
        // (512, 10, ActivationFunction::Softmax);
        // network.layers.push(first_layer);
        // network.layers.push(output_layer);
        let mut first_layer = Layer::new
        (2, 2, ActivationFunction::Relu);
        first_layer.weights[0] = vec!{0.1, -0.2};
        first_layer.weights[1] = vec!{0.3, 0.4};
        first_layer.biases = vec!{0.01, 0.02};
        let mut output_layer = Layer::new
        (2, 2, ActivationFunction::Softmax);
        output_layer.weights[0] = vec!{0.1, 0.2};
        output_layer.weights[1] = vec!{-0.1, 0.3};
        output_layer.biases = vec!{0.01, 0.02};
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
        let mut curr_input: Vec<f64> = input.clone();

        for curr_layer in self.layers.iter_mut() {
            curr_input = curr_layer.calculate(&curr_input);
        }
        return curr_input;
    }
    
    // actual_classification is the vector of 10 values containing 0s 
    // and 1 for the actual number the model was supposed to predict
    pub fn backpropagate(&mut self, output: &Vec<f64>, actual_classification: &Vec<f64>) -> f64 {
        let fake_zero = 1e-10;
        let loss: f64 = output
            .iter()
            .zip(actual_classification)
            .map(|(&output, &target)| -(target * (output.max(fake_zero).ln())))
            .sum::<f64>();        

        // Compute the gradient of the loss (cross-entrophy) bc we are using softmax
        // gradient is with respect to pre-softmax values
        // literally just sum of actual - predicted
        let output_gradient: Vec<f64> = output
            .iter()
            .zip(actual_classification)
            .map(|(&output, &target)| output - target) // Gradient for softmax with cross-entropy
            .collect();
        // println!("{:?}", output_gradient);

        // Perform backpropagation for each layer
        let mut gradient = output_gradient; // Initialize with the gradient of the output layer
        for i in (0..self.layers.len()).rev() {
            gradient = self.layers[i].backpropagate(gradient, self.learning_rate);
        }
        return loss;
    }

    // REAL TRAIN
    // pub fn train(&mut self, images_filename: &str, labels_filename: &str, iterations: usize) -> io::Result<()> {
    //     let mut image_reader = ImageReader::new(images_filename)?;
    //     let mut label_reader = LabelReader::new(labels_filename)?;
    
    //     let input_size = 28 * 28;
    //     let mut average_loss = 0.0;
    //     for iteration in 0..iterations {
    //         let image = image_reader.read_next_image().unwrap().unwrap();
    //         let label = label_reader.read_next_label().unwrap().unwrap();

    //         let prediction = self.calculate(&image);
    //         let mut actual_classification = vec![0.0; 10];
    //         actual_classification[label as usize] = 1.0;


    //         let loss = self.backpropagate(&prediction, &actual_classification);
    //         // println!();
    //         average_loss += loss;
    //         if iteration % 50 == 0 {
    //             average_loss /= 50.0;
    //             println!("Iteration {} completed, Average Loss over last 50 iterations = {}", iteration, average_loss);
                
    //             self.learning_rate *= 0.99;
    //             average_loss = 0.0;
    //         }
    //     }
    
    //     Ok(())
    // }

    // test train
    pub fn train(&mut self, images_filename: &str, labels_filename: &str, iterations: usize) -> io::Result<()> {
        let mut image_reader = ImageReader::new(images_filename)?;
        let mut label_reader = LabelReader::new(labels_filename)?;
    
        let input_size = 28 * 28;
        let mut average_loss = 0.0;

        for iteration in 0..iterations {
            let input = vec!{1.0, 2.0};

            let output = self.calculate(&input);
            for layer in &self.layers {
                println!("Layer output: {:?}", layer.input);
                println!("Layer z-values {:?}", layer.z_values);
            }
            println!("Output: {:?}", output);
            // let mut actual_classification = vec![0.0; 10];
            // actual_classification[label] = 1.0;
            let actual_classification = vec!{0.0, 1.0};

            let loss = self.backpropagate(&output, &actual_classification);
            println!("Loss: {}", loss);
            // println!();
            // average_loss += loss;
            // if iteration % 50 == 0 {
            //     average_loss /= 50.0;
            //     println!("Iteration {} completed, Average Loss over last 50 iterations = {}", iteration, average_loss);
                
            //     self.learning_rate *= 0.99;
            //     average_loss = 0.0;
            // }
        }
    
        Ok(())
    }
}

/// More positive weights are more green, and more negitive weights are more red. 
/// 
/// Weights close to 0 are more black.
fn weights_to_ppm(file_name: &str) {
    todo!();
}

struct ImageReader {
    reader: BufReader<File>,
    image_size: usize,
}

impl ImageReader {
    fn new(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        // Read the magic number
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        let magic_number = u32::from_be_bytes(buffer);
        if magic_number != 2051 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic number"));
        }
        // Read the number of images (not used directly here)
        reader.read_exact(&mut buffer)?;

        // Read the number of rows
        reader.read_exact(&mut buffer)?;
        let num_rows = u32::from_be_bytes(buffer) as usize;

        // Read the number of columns
        reader.read_exact(&mut buffer)?;
        let num_cols = u32::from_be_bytes(buffer) as usize;
        // Verify dimensions are 28x28
        if num_rows != 28 || num_cols != 28 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Expected 28x28 images"));
        }
        let image_size = num_rows * num_cols;

        Ok(ImageReader { reader, image_size })
    }

    fn read_next_image(&mut self) -> io::Result<Option<Vec<f64>>> {
        let mut buffer = vec![0u8; self.image_size];
        let bytes_read = self.reader.read(&mut buffer)?;

        if bytes_read == 0 {
            return Ok(None); // End of file
        }

        let image: Vec<f64> = buffer.iter().map(|&x| x as f64 / 255.0).collect();
        Ok(Some(image))
    }
}

struct LabelReader {
    reader: BufReader<File>,
    num_labels: usize,
}

impl LabelReader {
    fn new(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        // Read the header
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        let magic_number = u32::from_be_bytes(buffer);
        if magic_number != 2049 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic number"));
        }

        // Read the number of labels
        reader.read_exact(&mut buffer)?;
        let num_labels = u32::from_be_bytes(buffer) as usize;

        Ok(LabelReader { reader, num_labels })
    }

    fn read_next_label(&mut self) -> io::Result<Option<u8>> {
        let mut buffer = [0u8; 1];
        let bytes_read = self.reader.read(&mut buffer)?;

        if bytes_read == 0 {
            return Ok(None); // End of file
        }

        Ok(Some(buffer[0]))
    }
}