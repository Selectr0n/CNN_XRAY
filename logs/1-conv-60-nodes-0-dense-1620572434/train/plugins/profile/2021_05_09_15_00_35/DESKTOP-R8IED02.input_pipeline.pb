	?c?]K?5@?c?]K?5@!?c?]K?5@	????????????!??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?c?]K?5@?f??j+??A$(~??5@Y???x?&??*	fffff?W@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatesh??|???!Glz?F@)jM????1G"ʝ,LD@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX9??v???!?i?g?x0@)X9??v???1?i?g?x0@:Preprocessing2F
Iterator::ModelDio??ɔ?!0?????5@)%u???1;??L>/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?#??????!&|?=@)??0?*??1/Iag)@:Preprocessing2U
Iterator::Model::ParallelMapV2Ǻ???v?!F?*O??@)Ǻ???v?1F?*O??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj?q?????!?RTW?S@)??H?}m?1?????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF%u?k?!???%@)F%u?k?1???%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapI.?!????!????~?F@)Ǻ???V?1F?*O????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?f??j+???f??j+??!?f??j+??      ??!       "      ??!       *      ??!       2	$(~??5@$(~??5@!$(~??5@:      ??!       B      ??!       J	???x?&?????x?&??!???x?&??R      ??!       Z	???x?&?????x?&??!???x?&??JCPU_ONLYY??????b 