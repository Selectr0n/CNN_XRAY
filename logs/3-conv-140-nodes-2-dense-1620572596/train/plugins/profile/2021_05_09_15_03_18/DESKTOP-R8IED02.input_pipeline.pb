	???~?r_@???~?r_@!???~?r_@	?&s2????&s2???!?&s2???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???~?r_@??Q???A??9#Jk_@YvOjM??*	?????YM@2F
Iterator::Model??+e???!w
?βD@)e?X???1????x=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!V?`&??<@)y?&1???1??.??7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea??+e??!u\"?5@)Έ?????1??3??/@:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1?|?!??.??'@)y?&1?|?1??.??'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"??u????!???G1MM@)???_vOn?1z????6@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea??+ei?!u\"?@)a??+ei?1u\"?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!???cq@)?~j?t?h?1???cq@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???߾??!K|?V7@)??_?LU?1?45Қ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?&s2???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Q?????Q???!??Q???      ??!       "      ??!       *      ??!       2	??9#Jk_@??9#Jk_@!??9#Jk_@:      ??!       B      ??!       J	vOjM??vOjM??!vOjM??R      ??!       Z	vOjM??vOjM??!vOjM??JCPU_ONLYY?&s2???b 