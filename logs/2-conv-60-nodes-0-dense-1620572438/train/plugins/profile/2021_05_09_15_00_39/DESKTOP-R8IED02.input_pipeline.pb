	???lE@???lE@!???lE@	Ayִö?Ayִö?!Ayִö?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???lE@?}8gD??A46<?ZE@Y????ׁ??*	gffff&L@2F
Iterator::Model
ףp=
??!'?D?s?C@)%u???1??1:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	?^)ː?!f`$?1!=@)S?!?uq??1?????7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??0?*??!?Q????4@)"??u????1R?????.@:Preprocessing2U
Iterator::Model::ParallelMapV2?q?????!z<??m?+@)?q?????1z<??m?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX?5?;N??!?V??N@)	?^)?p?1f`$?1!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C??6j?!?>????@)-C??6j?1?>????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!?'?{P@)?~j?t?h?1?'?{P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?]K?=??!35I%??7@)?~j?t?X?1?'?{P@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ayִö?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}8gD???}8gD??!?}8gD??      ??!       "      ??!       *      ??!       2	46<?ZE@46<?ZE@!46<?ZE@:      ??!       B      ??!       J	????ׁ??????ׁ??!????ׁ??R      ??!       Z	????ׁ??????ׁ??!????ׁ??JCPU_ONLYYAyִö?b 