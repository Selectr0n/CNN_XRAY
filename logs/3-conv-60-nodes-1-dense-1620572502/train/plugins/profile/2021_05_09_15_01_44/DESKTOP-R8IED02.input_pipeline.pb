	aTR'?AG@aTR'?AG@!aTR'?AG@	???:Cľ????:Cľ?!???:Cľ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$aTR'?AG@0*??D??Au?V.G@YW[??재?*	gffff?K@2F
Iterator::ModelA??ǘ???!%0?e?D@)ŏ1w-!??1?v?,?|;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?St$????!	v?>@)9??v????1uK?7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??_vO??!?j??3@)	?^)ˀ?1P=?-@:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1?|?!?ґ=Q)@)y?&1?|?1?ґ=Q)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipm???{???!??'?{?M@)?J?4q?1???7a@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??H?}m?!?8/
@)??H?}m?1?8/
@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Le?!???ow?@)??_?Le?1???ow?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(??0??!|?-U>6@)?~j?t?X?1fkXY'?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???:Cľ?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0*??D??0*??D??!0*??D??      ??!       "      ??!       *      ??!       2	u?V.G@u?V.G@!u?V.G@:      ??!       B      ??!       J	W[??재?W[??재?!W[??재?R      ??!       Z	W[??재?W[??재?!W[??재?JCPU_ONLYY???:Cľ?b 