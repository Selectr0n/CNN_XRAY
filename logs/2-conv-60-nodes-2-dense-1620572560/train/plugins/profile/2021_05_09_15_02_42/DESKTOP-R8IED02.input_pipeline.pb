	V-zD@V-zD@!V-zD@	??I???????I?????!??I?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-zD@????Mb??Ax??#?dD@Y??ׁsF??*	      J@2F
Iterator::Model??&???!b'vb'6F@)??ܵ?|??1c'vb'?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?<,Ԛ???!;?;?<@)???????1??N??N6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ׁsF??!?؉??	3@)?<,Ԛ?}?1;?;?,@:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1?|?!?N??N?*@)y?&1?|?1?N??N?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??j+????!?؉???K@)_?Q?k?1wb'vb'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!;?;?@)?~j?t?h?1;?;?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Le?!      @)??_?Le?1      @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<?R??!c'vb'?4@)????MbP?1O??N????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??I?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Mb??????Mb??!????Mb??      ??!       "      ??!       *      ??!       2	x??#?dD@x??#?dD@!x??#?dD@:      ??!       B      ??!       J	??ׁsF????ׁsF??!??ׁsF??R      ??!       Z	??ׁsF????ׁsF??!??ׁsF??JCPU_ONLYY??I?????b 