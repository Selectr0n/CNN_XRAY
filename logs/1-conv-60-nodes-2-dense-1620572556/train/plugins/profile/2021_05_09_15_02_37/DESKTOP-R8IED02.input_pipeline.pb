	?W?2??9@?W?2??9@!?W?2??9@	??=?N0????=?N0??!??=?N0??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?W?2??9@ё\?C???A`??"۹9@Y?j+??ݣ?*	?????L@2F
Iterator::Model???????!?k[??D@)?q??????1??#??;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9??v???!pMc??;@)???<,Ԋ?1?]?BO7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?(??0??!?\???5@)/n????1?(?Q/@:Preprocessing2U
Iterator::Model::ParallelMapV2ŏ1w-!?!??"??+@)ŏ1w-!?1??"??+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL7?A`???!????\M@)ŏ1w-!o?1??"??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!??e9?@)y?&1?l?1??e9?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!b?(?@)a2U0*?c?1b?(?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???߾??!?3?Ñ`8@)Ǻ???V?1G????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??=?N0??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ё\?C???ё\?C???!ё\?C???      ??!       "      ??!       *      ??!       2	`??"۹9@`??"۹9@!`??"۹9@:      ??!       B      ??!       J	?j+??ݣ??j+??ݣ?!?j+??ݣ?R      ??!       Z	?j+??ݣ??j+??ݣ?!?j+??ݣ?JCPU_ONLYY??=?N0??b 