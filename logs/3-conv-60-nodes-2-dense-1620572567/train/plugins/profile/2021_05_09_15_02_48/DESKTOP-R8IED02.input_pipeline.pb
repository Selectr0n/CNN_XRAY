	??1??UF@??1??UF@!??1??UF@	?I;?????I;????!?I;????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??1??UF@?o_???A???S?=F@Y?^)?Ǫ?*	     ?R@2F
Iterator::Model?Q?????!UUUUUUG@)??e?c]??1xwwwwwB@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??d?`T??!??????7@)%u???1??????3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2U0*???!??????4@)S?!?uq??1??????1@:Preprocessing2U
Iterator::Model::ParallelMapV2?<,Ԛ?}?!wwwwww#@)?<,Ԛ?}?1wwwwww#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{?G?z??!??????J@)U???N@s?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C??6j?!@)-C??6j?1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!??????@)HP?s?b?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn????!!""""":@)_?Q?[?1""""""@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9I;????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?o_????o_???!?o_???      ??!       "      ??!       *      ??!       2	???S?=F@???S?=F@!???S?=F@:      ??!       B      ??!       J	?^)?Ǫ??^)?Ǫ?!?^)?Ǫ?R      ??!       Z	?^)?Ǫ??^)?Ǫ?!?^)?Ǫ?JCPU_ONLYYI;????b 