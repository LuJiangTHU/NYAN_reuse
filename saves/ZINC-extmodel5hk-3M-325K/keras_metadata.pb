
�root"_tf_keras_model*�{"name": "vae", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "VAE", "config": {}, "shared_object_id": 0, "build_input_shape": {"class_name": "__tuple__", "items": [{"class_name": "TensorShape", "items": [64, 65, 16]}, {"class_name": "TensorShape", "items": [64, 65, 65]}, {"class_name": "TensorShape", "items": [64, 65, 65, 5]}]}, "is_graph_network": false, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "__tuple__", "items": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 65, 16]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 65, 65]}, "int64", "input_2"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 65, 65, 5]}, "float32", "input_3"]}]}], {}]}, "save_spec": {"class_name": "__tuple__", "items": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 65, 16]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 65, 65]}, "int64", "input_2"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 65, 65, 5]}, "float32", "input_3"]}]}, "keras_version": "2.10.0", "backend": "tensorflow", "model_config": {"class_name": "VAE", "config": {}}, "training_config": {"loss": "combined_loss", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "binary_acc", "dtype": "float32", "fn": "binary_acc"}, "shared_object_id": 1}, {"class_name": "MeanMetricWrapper", "config": {"name": "regression_mae", "dtype": "float32", "fn": "regression_mae"}, "shared_object_id": 2}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": {"class_name": "InverseTimeDecay", "config": {"initial_learning_rate": 0.0002, "decay_steps": 300000, "decay_rate": 0.35, "staircase": false, "name": null}, "shared_object_id": 3}, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-08, "amsgrad": false}}}}2
�
�	root.decoder"_tf_keras_model*�{"name": "nyan_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "NyanDecoder", "config": {}, "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [64, 64]}, "is_graph_network": false, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 64]}, "float32", "input_1"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 64]}, "float32", "input_1"]}, "keras_version": "2.10.0", "backend": "tensorflow", "model_config": {"class_name": "NyanDecoder", "config": {}}}2
�?root.encoder.masking"_tf_keras_layer*�{"name": "graph_masking", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "GraphMasking", "config": {"name": "graph_masking", "trainable": true, "dtype": "float32"}, "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 16]}}2
�	@root.encoder.precondition"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 7}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 15]}}2
�Aroot.encoder.graphconv1"_tf_keras_layer*�{"name": "ecc_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "ECCConv", "config": {"name": "ecc_conv", "trainable": true, "dtype": "float32", "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 12}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 32, "kernel_network": null, "root": true}, "shared_object_id": 15, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 65, 16]}, {"class_name": "TensorShape", "items": [64, 65, 65]}, {"class_name": "TensorShape", "items": [64, 65, 65, 5]}]}2
�Broot.encoder.graphconv2"_tf_keras_layer*�{"name": "ecc_conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "ECCConv", "config": {"name": "ecc_conv_1", "trainable": true, "dtype": "float32", "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 16}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 32, "kernel_network": null, "root": true}, "shared_object_id": 19, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 65, 32]}, {"class_name": "TensorShape", "items": [64, 65, 65]}, {"class_name": "TensorShape", "items": [64, 65, 65, 5]}]}2
�Croot.encoder.graphconv3"_tf_keras_layer*�{"name": "ecc_conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "ECCConv", "config": {"name": "ecc_conv_2", "trainable": true, "dtype": "float32", "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 20}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "channels": 32, "kernel_network": null, "root": true}, "shared_object_id": 23, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 65, 32]}, {"class_name": "TensorShape", "items": [64, 65, 65]}, {"class_name": "TensorShape", "items": [64, 65, 65, 5]}]}2
�Droot.encoder.pool1"_tf_keras_layer*�{"name": "global_sum_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "GlobalSumPool", "config": {"name": "global_sum_pool", "trainable": true, "dtype": "float32"}, "shared_object_id": 24, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 32]}}2
�	Eroot.encoder.dense1"_tf_keras_layer*�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 25}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 32]}}2
�Froot.encoder.flatten"_tf_keras_layer*�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
�	Groot.encoder.dense2"_tf_keras_layer*�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 32}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
�Hroot.encoder.z_mean"_tf_keras_layer*�{"name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
�Iroot.encoder.z_log_var"_tf_keras_layer*�{"name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
�Jroot.encoder.latent_z"_tf_keras_layer*�{"name": "sampling", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "shared_object_id": 45, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 64]}, {"class_name": "TensorShape", "items": [64, 64]}, {"class_name": "TensorShape", "items": []}]}2
�	Rroot.decoder.dense3"_tf_keras_layer*�{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 46}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 64]}}2
�Sroot.decoder.fingerprint"_tf_keras_layer*�{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 679, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 53, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
�Troot.decoder.regression"_tf_keras_layer*�{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1613, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
�[root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 59}2
�\root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "binary_acc", "dtype": "float32", "config": {"name": "binary_acc", "dtype": "float32", "fn": "binary_acc"}, "shared_object_id": 1}2
�]root.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "regression_mae", "dtype": "float32", "config": {"name": "regression_mae", "dtype": "float32", "fn": "regression_mae"}, "shared_object_id": 2}2
�^root.keras_api.metrics.3"_tf_keras_metric*�{"class_name": "Mean", "name": "kl_loss", "dtype": "float32", "config": {"name": "kl_loss", "dtype": "float32"}, "shared_object_id": 60}2
�_root.keras_api.metrics.4"_tf_keras_metric*�{"class_name": "Mean", "name": "kl_loss_beta", "dtype": "float32", "config": {"name": "kl_loss_beta", "dtype": "float32"}, "shared_object_id": 61}2
�y$root.encoder.precondition.activation"_tf_keras_layer*�{"name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 16]}}2
��"root.encoder.graphconv1.activation"_tf_keras_layer*�{"name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 32]}}2
��"root.encoder.graphconv2.activation"_tf_keras_layer*�{"name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 16, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 32]}}2
��"root.encoder.graphconv3.activation"_tf_keras_layer*�{"name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 32]}}2
��root.encoder.dense1.activation"_tf_keras_layer*�{"name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 25, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
��root.encoder.dense2.activation"_tf_keras_layer*�{"name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 32, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
��root.decoder.dense3.activation"_tf_keras_layer*�{"name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.05000000074505806}, "shared_object_id": 46, "build_input_shape": {"class_name": "TensorShape", "items": [64, 256]}}2
��/root.encoder.graphconv1.kernel_network_layers.0"_tf_keras_layer*�{"name": "FGN_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "FGN_out", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 62}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 63}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 64, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 65, 5]}}2
��/root.encoder.graphconv2.kernel_network_layers.0"_tf_keras_layer*�{"name": "FGN_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "FGN_out", "trainable": true, "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 66}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 67}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 68, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 65, 5]}}2
��/root.encoder.graphconv3.kernel_network_layers.0"_tf_keras_layer*�{"name": "FGN_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "FGN_out", "trainable": true, "dtype": "float32", "units": 1024, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 70}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 71}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 72, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 65, 65, 5]}}2