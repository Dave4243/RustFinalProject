
use std::{error::Error, io, process};
use na::{U2, U3, Dynamic, ArrayStorage, VecStorage, Matrix};
use std::f64::consts::E;
use std::vec::Vec;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead};
use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

// std::f64::consts::E;

pub struct UbyteFile {
    sizes: Vec<i32>;
    data: Vec<u8>;
}

impl UbyteFile {
    fn new(f: &str) -> Result<UbyteFile, Error> {
        let mut file = File::open(input_file)?;
        let mut vec = Vec::new();
        file.read_to_end(vec)?;
        let mut r = Cursor::new(vec);

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        let magic_number = r.read_i32::<BigEndian>()?;
        match magic_number {
            // 2049 => {
            //     sizes.push(r.read_i32::<BigEndian>()?);
            // }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!();
        }
        r.read_to_end(data)?;
        Ok(UbyteFile{sizes, data})
    }
}

// pub mod model {
// }
static B:f64 = 1.0;

trait ActivationFunction {
    fn calculate(input:f64, min:f64, max:f64) -> f64;
}

pub struct Sigmoid {}

impl ActivationFunction for Sigmoid {
    fn calculate(input:f64, min:f64, max:f64) -> f64 {
        1.0/(E.powf(-input * B))
    }
}

/// Describes 1D vector layer. 
/// 
/// Each layer contains the weights that relate it to its previous layer.
/// 
/// The matrix multiplation of the previous layer's output and this layer's weights matrix results in this layers output 
/// 
/// Also contains optional biases, there is a bias for each node in the layer. This repesents a flat amount to add to the activation sum after accounting for weights
pub struct Layer {
    // _nodes: [f64; layer_size],

    _size: usize,
    // _prev_layer_size: Option<usize>,

    _weights: Vec<Vec<f64>>, // layer_size,

    /// The bias for each node. Should be added to the sum of 
    _biases: Option<Vec<f64>>,
    // _data: [f64; N] = [0.0; N];
    // _weights: [f64; N] = [0.0; N];

    // Layer()

    // Layer 

}

impl Layer { 
    pub fn new() -> Self {
        todo!()
    }

    pub fn calculate(prevous_layer_val: Vec<f64>) -> Vec<f64> {
        todo!()
    }
}

pub struct Network<'a> /*<I: Iterator<Item=[[f64; 28]; 28]>>*/ {
    // _num_layers: usize,
    // _input: [[f64;28]; 28],
    // _output: [[f64;28];28],

    _input_layer: &'a Layer,
    _layers: Vec<Layer>,
    _output_layer: &'a Layer,


    // vec<[[f64;]]>
}

impl<'a> Network<'a> {
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
        //let mut file = File::create("output.txt")?;
        let file = OpenOptions::new().append(true).open(file_name).expect("Unable to open file");
        file.write_all(b"{}", 3021 as i32)?;
        
        Ok(())
    }

    pub fn new() -> Self {
        todo!()
    }

    pub fn add_layer(self: &mut Self, layer: Layer) -> Result<(), ()> {
        match self._layers.last() {
            // This code might be wrong lol
            Some(x) if x._size != layer._weights.len() => {
                return Err(());
            }    
            None => {}
            _ => {}
        }

        self._layers.push(layer);

        Ok(())
        
        // todo!()
        // Ok(())
    }

    pub fn calculate(input: Vec<f32>) -> [f64; 10] {
        todo!()
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
    let input = &UbyteFile::new(&str);
    let image_shape = (input.sizes[1] * input.sizes[2]) as usize;
    let index = image_shape*row;
    let read_file_vec: Vec<Vec<f64>> = Vec::new();
    for i in 0..input.sizes[1] {
        read_file_vec.push(Vec::new());
        for j in 0..input.sizes[2] {
            let data: f64 = input.data[i] as f64 / 255;
            read_file_vec[i].push(data);
        }
    }
    Ok(read_file_vec)
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
/// Panic if all input values are 0
///
/// Returns the normalized array. 
fn normalize_output(output: &[f64; 10]) -> [f64;10] {
    todo!();
}

// fn([[f64;28]; 28]);

