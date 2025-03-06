import argparse
import os, time, math
import numpy as np
from glob import glob
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# --- Command-Line Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="Total training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for optimizers")
parser.add_argument("--b1", type=float, default=0.5, help="Beta1 for Adam")
parser.add_argument("--b2", type=float, default=0.9, help="Beta2 for Adam")
parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of noise vector")
parser.add_argument("--img_size", type=int, default=64, help="Image width/height")
parser.add_argument("--channels", type=int, default=1, help="Image channels: 1 for grayscale, 3 for RGB")
parser.add_argument("--n_critic", type=int, default=5, help="Discriminator updates per generator update")
parser.add_argument("--sample_interval", type=int, default=500, help="Steps between saving generated samples")
parser.add_argument("--data_dir", type=str, default="data/your_dataset", help="Directory for input images")
opt = parser.parse_args()
print("Options:", opt)
img_shape = (opt.img_size, opt.img_size, opt.channels)

# --- Dataset Loading ---
def load_dataset(data_dir, target_size):
    """Load and preprocess images from data_dir."""
    jpg_files = glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    png_files = glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    files = jpg_files + png_files
    print("Found {} images in {}".format(len(files), data_dir))
    images = []
    for path in files:
        try:
            img = Image.open(path).convert('L' if opt.channels == 1 else 'RGB')
            img = img.resize((target_size, target_size), Image.ANTIALIAS)
            arr = np.array(img, dtype=np.float32)
            if opt.channels == 1 and arr.ndim == 2:
                arr = arr[..., np.newaxis]
            # Normalize image to [-1, 1]
            images.append(arr / 127.5 - 1.0)
        except Exception as e:
            print("Error loading {}: {}".format(path, e))
    return np.array(images)

# --- Model Layers ---
def dense_layer(inputs, units, scope_name, activation=None):
    """Create a fully connected layer."""
    with tf.variable_scope(scope_name):
        return tf.layers.dense(inputs, units, activation=activation)

# --- Generator ---
def build_generator(z, batch_size, latent_dim, output_shape):
    """Map latent vector z to an image with several dense layers."""
    with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
        net = dense_layer(z, 128, "dense1", activation=tf.nn.leaky_relu)
        net = tf.layers.batch_normalization(net, training=True)
        net = dense_layer(net, 256, "dense2", activation=tf.nn.leaky_relu)
        net = tf.layers.batch_normalization(net, training=True)
        net = dense_layer(net, 512, "dense3", activation=tf.nn.leaky_relu)
        net = tf.layers.batch_normalization(net, training=True)
        net = dense_layer(net, 1024, "dense4", activation=tf.nn.leaky_relu)
        net = tf.layers.batch_normalization(net, training=True)
        net = dense_layer(net, np.prod(output_shape), "dense5", activation=tf.nn.tanh)
        return tf.reshape(net, [batch_size] + list(output_shape))

# --- Discriminator ---
def build_discriminator(img, reuse=False):
    """Classify images using dense layers."""
    with tf.variable_scope("Discriminator", reuse=reuse):
        flat = tf.layers.flatten(img)
        net = dense_layer(flat, 512, "dense1", activation=tf.nn.leaky_relu)
        net = dense_layer(net, 256, "dense2", activation=tf.nn.leaky_relu)
        return dense_layer(net, 1, "dense3")

# --- Gradient Penalty ---
def compute_gradient_penalty(D, real, fake, batch_size):
    """Enforce Lipschitz constraint via gradient penalty."""
    alpha = tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolates = alpha * real + (1 - alpha) * fake
    D_interpolates = D(interpolates, reuse=True)
    gradients = tf.gradients(D_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    return tf.reduce_mean((slopes - 1.) ** 2)

# --- Image Grid Utilities ---
def merge_images(images, grid_size):
    """Merge images into a grid."""
    h, w = images.shape[1], images.shape[2]
    merged = np.zeros((h * grid_size[0], w * grid_size[1]))
    for idx, img in enumerate(images):
        row, col = idx // grid_size[1], idx % grid_size[1]
        merged[row*h: row*h+h, col*w: col*w+w] = np.squeeze(img)
    return merged

def save_image_grid(images, path):
    """Save a grid of images to disk."""
    grid_side = int(math.sqrt(images.shape[0]))
    merged = merge_images((images + 1.0) / 2.0, (grid_side, grid_side))
    im = Image.fromarray((merged * 255).astype(np.uint8))
    im.save(path)

# --- Build Computational Graph ---
BATCH_SIZE = opt.batch_size
real_imgs_ph = tf.placeholder(tf.float32, [BATCH_SIZE] + list(img_shape), name="real_imgs")
z_ph = tf.placeholder(tf.float32, [BATCH_SIZE, opt.latent_dim], name="z")
fake_imgs = build_generator(z_ph, BATCH_SIZE, opt.latent_dim, img_shape)
D_real = build_discriminator(real_imgs_ph, reuse=False)
D_fake = build_discriminator(fake_imgs, reuse=True)

# WGAN losses with gradient penalty.
d_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
gp = compute_gradient_penalty(build_discriminator, real_imgs_ph, fake_imgs, BATCH_SIZE)
lambda_gp = 10
d_loss_total = d_loss + lambda_gp * gp
g_loss = -tf.reduce_mean(D_fake)

# Optimizers.
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
optimizer_D = tf.train.AdamOptimizer(opt.lr, beta1=opt.b1, beta2=opt.b2).minimize(d_loss_total, var_list=disc_vars)
optimizer_G = tf.train.AdamOptimizer(opt.lr, beta1=opt.b1, beta2=opt.b2).minimize(g_loss, var_list=gen_vars)

# Setup checkpoint and sample directories.
saver = tf.train.Saver()
ckpt_dir = "checkpoints"
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
sample_dir = "samples"
if not os.path.exists(sample_dir): os.makedirs(sample_dir)

# --- Training Loop ---
def train():
    """Run the training process."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    dataset = load_dataset(opt.data_dir, opt.img_size)
    total_imgs = dataset.shape[0]
    if total_imgs < BATCH_SIZE:
        raise ValueError("Dataset size ({}) is smaller than batch size ({}).".format(total_imgs, BATCH_SIZE))
    total_batches = total_imgs // BATCH_SIZE
    print("Loaded {} images.".format(total_imgs))
    
    step = 0
    start_time = time.time()
    for epoch in range(opt.n_epochs):
        for i in range(total_batches):
            batch_real = dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            noise = np.random.normal(0, 1, (BATCH_SIZE, opt.latent_dim)).astype(np.float32)
            # Update discriminator multiple times.
            for _ in range(opt.n_critic):
                _, d_loss_val = sess.run([optimizer_D, d_loss_total],
                                         feed_dict={real_imgs_ph: batch_real, z_ph: noise})
            # Update generator.
            _, g_loss_val = sess.run([optimizer_G, g_loss], feed_dict={z_ph: noise})
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print("[Epoch {}/{}] [Batch {}/{}] [D: {:.4f}] [G: {:.4f}] [Time: {:.2f}s]".format(
                    epoch, opt.n_epochs, i, total_batches, d_loss_val, g_loss_val, elapsed))
            # Save generated samples.
            if step % opt.sample_interval == 0:
                sample_noise = np.random.normal(0, 1, (BATCH_SIZE, opt.latent_dim)).astype(np.float32)
                samples = sess.run(fake_imgs, feed_dict={z_ph: sample_noise})
                sample_path = os.path.join(sample_dir, "{}.png".format(step))
                save_image_grid(samples, sample_path)
                print("Saved samples to", sample_path)
            # Save checkpoints.
            if step % 1000 == 0 and step > 0:
                saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=step)
                print("Saved checkpoint at step", step)
            step += 1

if __name__ == "__main__":
    train()