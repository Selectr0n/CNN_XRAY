	??JY?_@??JY?_@!??JY?_@	?^c??????^c?????!?^c?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??JY?_@?2ı.n??A~8gD_@Y333333??*	gffff&L@2F
Iterator::ModelM??St$??!f[?20D@)X9??v???1??G??;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"??u????!R?????>@)?<,Ԛ???1?Ok???9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??_vO??!?dV??.3@)????Mb??1o?ޒOk,@:Preprocessing2U
Iterator::Model::ParallelMapV2?ZӼ?}?!??%??8)@)?ZӼ?}?1??%??8)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip7?[ A??!?????M@)????Mbp?1o?ޒOk@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ???f?!??????@)Ǻ???f?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!????x@)??_?Le?1????x@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????!H?*?36@)_?Q?[?1?b=?(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?^c?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?2ı.n???2ı.n??!?2ı.n??      ??!       "      ??!       *      ??!       2	~8gD_@~8gD_@!~8gD_@:      ??!       B      ??!       J	333333??333333??!333333??R      ??!       Z	333333??333333??!333333??JCPU_ONLYY?^c?????b 