	?????$@?????$@!?????$@	?jEdq&???jEdq&??!?jEdq&??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????$@/n????AgDio?%$@Y???3???*	?????YM@2F
Iterator::Modelݵ?|г??!??3?@aE@)?Q?????1B?T??=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ǘ????!????;@)_?Qڋ?1x???,+7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??@??ǈ?!?I?o ?4@)??y?):??1????R.@:Preprocessing2U
Iterator::Model::ParallelMapV2ŏ1w-!?!?`&???)@)ŏ1w-!?1?`&???)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?J?4??!+???L@)??H?}m?1%??q?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C??6j?!??-eH?@)-C??6j?1??-eH?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!?45Қ?@)??_?Le?1?45Қ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaplxz?,C??!?^f?7@)_?Q?[?1x???,+@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?jEdq&??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/n????/n????!/n????      ??!       "      ??!       *      ??!       2	gDio?%$@gDio?%$@!gDio?%$@:      ??!       B      ??!       J	???3??????3???!???3???R      ??!       Z	???3??????3???!???3???JCPU_ONLYY?jEdq&??b 