"""An online triplet loss implementation method"""


"""
Triplet Loss: 

    L = max(dis(anchor, positive) - dis(anchor, negative) + margin, 0)

"""
import tensorflow as tf
import numpy as np 

# import tensorflow.keras as keras


def _pairwise_euclidean_distance(embeddings, squared=False):
    """Compute L2 distances between each instance in embedding space;
    Args: 
        embeddings: 2D FloatTensor of shape (batch_size, embedding_size)
        squared: bool. If True, returns the squared L2 norm, else, return Euclidean distances.
    
    Returns:
        distances: 2D FloatTensor of shape (batch_size, batch_size)

    """


    # Dot product between each instance:
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # L2 norm squared
    square_norm = tf.linalg.diag_part(dot_product)

    # squared distances 
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        distances = distances * (1.0 - mask)
    
    return distances


def _pairwise_cosine_distance(embeddings):
    """Compute Cosine distances between each instance in embedding space

    Args:
        embeddings: 2D FloatTensor of shape (batch_size, embedding_size)

    Returns:
        distances: 2D FloatTensor of shape (batch_size, batch_size)
    """
    # TODO


    pass


def _get_positive_mask(labels):
    """Return a 2D matrix where mask [a, p] is True iff a and p are distince and have the same label.

    Args:
        labels: tf.int32/long Tensor with shape [batch_size, ]
        

    Returns:
        mask: tf.bool Tensor with shape [batch_size, batch_size]    

    """
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_negative_mask(labels):
    """Return a 2D matrix where mask [a, n] is True iff a and n are distince and have the distinct label.

    Args:
        labels: tf.int32/long Tensor with shape [batch_size, ]
        

    Returns:
        mask: tf.bool Tensor with shape [batch_size, batch_size]    

    """
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)

    return mask



def _get_triplet_mask(labels):
    """Return 3D mask where mask [a, p, n] is True iff the triple (a, p, n) is valid.

    A triple (i, j, k) is valid if:
        -- i, j, k are distinct;
        -- label[i] == label[j] and label[i] != label[j]

    Args:
        labels: tf.int32/tf.long Tensor with shape [batch_size,]
        
    Returns:
        mask: tf.bool Tensor with shape [batch_size, batch_size, batch_size]
    """

    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    labesl_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(labels_equal, 2)
    i_equal_k = tf.expand_dims(labels_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):

    pairwise_dist = _pairwise_euclidean_distance(embeddings, squared=squared)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)

    triplet_loss = tf.multiply(mask, triplet_loss)

    triplet_loss = tf.maximum(triplet_loss, 0.0)
    valid_triplet = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplet)
    num_valid_triplets = tf.reduce_sum(mask)

    fraction_positive_triplets = num_positive_triplets/ (num_valid_triplets + 1e-16)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin=0.1, squared=False, soft_margin=False):
    """Implementation of Batch-Hard Triplet Mining from `In Defence of the Triplet Loss for 
    Person Re-Identification` (Hermans et al. 2017).

    Build triplets from batch of embeddings: for each anchor, mine the hardest positive and negative loss

    Args:
        labels: int32 tensor of shape [batch_size,]
        embeddings: FloatTensor of shape [batch_size, embedding_size]
        margin: scalar for triplet loss, in paper, the author choose among {0.1, 0.2, 0.5, 1.0}
        squared: bool, if True, use squared euclidien distance.
        soft_margin: bool, if True, use soft margin, overwrites margin argument.
                     Change from a ReLU operation to Softplus operation.

    Returns:
        loss: a scalar tensor containing the triplet loss value.

    """

    pairwise_dist = _pairwise_euclidean_distance(embeddings, squared=squared)
    mask_positive = tf.to_float(_get_positive_mask(labels))

    anchor_positive_dist = tf.multiply(mask_positive, pairwise_dist)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    mask_negative = tf.to_float(_get_negative_mask(labels))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_negative)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))
    if not soft_margin:
        loss = tf.maximum(hardest_positive_dist + margin - hardest_negative_dist, 0.0)
        loss = tf.reduce_mean(loss)
    else:
        loss = tf.nn.softplus(margin + hardest_positive_dist - hardest_negative_dist)
    return loss 


if __name__ == "__main__":
    tf.enable_eager_execution()

    # test_tensor = tf.random_normal((4,3))
    # print(test_tensor)

    # dot_product = tf.matmul(test_tensor, tf.transpose(test_tensor))
    # print(dot_product)

    # square_norm = tf.diag_part(dot_product)
    # distance = tf.expand_dims(square_norm, 0) - 2 * dot_product + tf.expand_dims(square_norm, 1)
    # print(distance)
    test_label_tensor = np.random.randint(5, size=(10, 10))
    # print(test_label_tensor)
    test_label_tensor = tf.convert_to_tensor(test_label_tensor, dtype=tf.int32)
    haha = tf.expand_dims(test_label_tensor, 2)
    print(test_label_tensor)
    print(haha)


    # print(test_label_tensor)
    # indices_equal = tf.cast(tf.eye(tf.shape(test_label_tensor)[0]), tf.bool)

    # print(indices_equal)