	??ܵV@??ܵV@!??ܵV@	'?Ԋ溦?'?Ԋ溦?!'?Ԋ溦?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ܵV@??x?&1??A\???(?U@YM?J???*	?????LM@2F
Iterator::Model??0?*??!1(???"D@)?Q?????1???=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL7?A`???!l???0(<@)F%u???1??늍?6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?~j?t???!??3!Rz4@)/n????1?:??.@:Preprocessing2U
Iterator::Model::ParallelMapV2?HP?x?!%?|]??$@)?HP?x?1%?|]??$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Q?????!???M@)/n??r?1?:??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!??늍?@)F%u?k?1??늍?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C??6j?!?Y??@)-C??6j?1?Y??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?!??u???!?+6?8@)?J?4a?1#????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9&?Ԋ溦?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??x?&1????x?&1??!??x?&1??      ??!       "      ??!       *      ??!       2	\???(?U@\???(?U@!\???(?U@:      ??!       B      ??!       J	M?J???M?J???!M?J???R      ??!       Z	M?J???M?J???!M?J???JCPU_ONLYY&?Ԋ溦?b 