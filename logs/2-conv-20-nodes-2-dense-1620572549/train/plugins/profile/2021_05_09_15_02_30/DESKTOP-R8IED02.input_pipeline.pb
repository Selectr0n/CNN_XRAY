	"lxz?|1@"lxz?|1@!"lxz?|1@	7?ɵ0???7?ɵ0???!7?ɵ0???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"lxz?|1@333333??A?;NёL1@Y?ݓ??Z??*	     ?L@2F
Iterator::Model?b?=y??!q?}?D@)?? ?rh??11??t?=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????Mb??!?}?<@)S?!?uq??1$???>?7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??0?*??!;??,??4@)/n????1	?#???.@:Preprocessing2U
Iterator::Model::ParallelMapV2lxz?,C|?!^Cy?5(@)lxz?,C|?1^Cy?5(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipm???{???!??>??M@)ŏ1w-!o?1??????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?h?!?P^Cy@)?~j?t?h?1?P^Cy@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!?#???>@)??_?Le?1?#???>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?]K?=??!UUUUUU7@)?~j?t?X?1?P^Cy@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no97?ɵ0???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	333333??333333??!333333??      ??!       "      ??!       *      ??!       2	?;NёL1@?;NёL1@!?;NёL1@:      ??!       B      ??!       J	?ݓ??Z???ݓ??Z??!?ݓ??Z??R      ??!       Z	?ݓ??Z???ݓ??Z??!?ݓ??Z??JCPU_ONLYY7?ɵ0???b 