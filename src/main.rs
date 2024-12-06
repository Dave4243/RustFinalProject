mod model;

use crate::model::Network;

// use model;

fn main() {
    let mut network = Network::new();
    network.train(
        r"data\t10k-images.idx3-ubyte"
    , r"data\t10k-labels.idx1-ubyte", 1000);
}
