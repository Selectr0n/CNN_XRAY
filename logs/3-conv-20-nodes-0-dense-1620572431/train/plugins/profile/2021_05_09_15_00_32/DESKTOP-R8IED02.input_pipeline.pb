	?٬?\-1@?٬?\-1@!?٬?\-1@	Ӈ ?g??Ӈ ?g??!Ӈ ?g??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?٬?\-1@???h o??A??b??0@Y?D???J??*	33333SQ@2U
Iterator::Model::ParallelMapV2??_?L??!?x??>@)??_?L??1?x??>@:Preprocessing2F
Iterator::Model?Q?????!??~?@I@)?ZӼ???1e??M?}4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!?B9??8@)_?Qڋ?1??-??3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatetF??_??!֯??+1@)?J?4??1>???>(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip o?ŏ??!j?J?Z?H@)	?^)?p?10?L?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!?ΐ??3@)y?&1?l?1?ΐ??3@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?k?!??-??@)_?Q?k?1??-??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u???!?0?03@)??_?LU?1?x????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ӈ ?g??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???h o?????h o??!???h o??      ??!       "      ??!       *      ??!       2	??b??0@??b??0@!??b??0@:      ??!       B      ??!       J	?D???J???D???J??!?D???J??R      ??!       Z	?D???J???D???J??!?D???J??JCPU_ONLYYӇ ?g??b 