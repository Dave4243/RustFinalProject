mod model;

use crate::model::Network;

fn main() {
    let mut network = Network::new();
    let _ = network.train(
        r"data\train-images.idx3-ubyte"
    , r"data\train-labels.idx1-ubyte", 10000);
    network.write_to_file("network_file");
}
