	S??:q1@S??:q1@!S??:q1@	|3?Mdm??|3?Mdm??!|3?Mdm??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$S??:q1@??Q????A???V?/1@Y??#?????*	?????La@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?f??j+??!?wK??I?@)?0?*???1vp??=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??????!DZ/`?S6@)B>?٬???1{ۜ?r4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?|a2U??!}???G@)ˡE?????1>П-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!?rh????!????niT@)46<???1t??3?q*@:Preprocessing2F
Iterator::Model?
F%u??!???DZ2@)r??????1?}???)@:Preprocessing2U
Iterator::Model::ParallelMapV2? ?	??!?j?A@)? ?	??1?j?A@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea??+ei?!1;8?H?@)a??+ei?11;8?H?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!??E5???)??_?Le?1??E5???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9|3?Mdm??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Q??????Q????!??Q????      ??!       "      ??!       *      ??!       2	???V?/1@???V?/1@!???V?/1@:      ??!       B      ??!       J	??#???????#?????!??#?????R      ??!       Z	??#???????#?????!??#?????JCPU_ONLYY|3?Mdm??b 