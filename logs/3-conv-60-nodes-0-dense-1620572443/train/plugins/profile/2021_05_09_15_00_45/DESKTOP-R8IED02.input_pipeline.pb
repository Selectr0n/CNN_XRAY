	Dio???G@Dio???G@!Dio???G@	??ұ????ұ??!??ұ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Dio???G@-!?lV??A??m4??G@Y46<???*	33333sJ@2F
Iterator::Model??_?L??!wj??C@)???QI??1??c:;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK?=?U??!??x?(?<@)9??v????1???8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??0?*??!u.?eN6@)??y?):??1?W???0@:Preprocessing2U
Iterator::Model::ParallelMapV29??v??z?!???(@)9??v??z?1???(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8gDio??!????WN@)??H?}m?1#8̺?8@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????g?!f]O??@)?????g?1f]O??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!???,d@)HP?s?b?1???,d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u???! ?????8@)Ǻ???V?1H??	,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??ұ??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-!?lV??-!?lV??!-!?lV??      ??!       "      ??!       *      ??!       2	??m4??G@??m4??G@!??m4??G@:      ??!       B      ??!       J	46<???46<???!46<???R      ??!       Z	46<???46<???!46<???JCPU_ONLYY??ұ??b 