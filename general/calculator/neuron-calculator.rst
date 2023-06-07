.. _neuron_calculator:

Neuron Calculator
=================

.. raw:: html

            <script>
                require.config({
                   paths: {
                        mathjs: 'https://cdnjs.cloudflare.com/ajax/libs/mathjs/11.8.0/math.min'
                        }
                    });
            </script>


        <div class="container">

            <div id="neuron-calculator-select" class="form-group row">
                <label for="formSelect" class="col-sm-3 col-form-label fw-bold text-end"> Select the Calculator</label>
                <div class="col-sm-8">

                    <select class="form-control" id="formSelect" name="formSelect">
                        <option value="compute-core-form"> Neuron Cores needed for LLM Inference</option>
                    </select>

                </div>
            </div>



        <form id="compute-core-form" method="get" onsubmit="submitComputeCoreForm(); return false;">

            <h2> Number of Neuron Cores needed for LLM Inference</h2>
            
                 <div class="form-group row">
                    <label for="model-select" class="col-sm-3 col-form-label fw-bold text-end"> Model: </label>
                     <div class="col-sm-8">

                    <select class="form-control" id="model-select" style="width:100%;">
                        <option value="custom-llm-model"> Custom LLM Model </option>

                        <optgroup label="Sample Model Configuration" class="font-weight-bold" >
                            <option value="opt-66b"> opt-66b </option>
                        </optgroup>
                    </select>
                </div>
            </div>

                 <div class="form-group row">
                    <label for="instance-type" class="col-sm-3 col-form-label fw-bold text-end"> Instance Type: </label>
                        <div class="col-sm-8">
                    <select class="form-control" id="instance-type" >
                        <option value="Inf2"> Inf2 </option>
                        <option value="Trn1"> Trn1 </option>
                    </select>
                </div>
            </div>


               <div class="form-group row">
                    <label for="data-type" class="col-sm-3 col-form-label fw-bold text-end"> Data Type: </label>
                        <div class="col-sm-8">
                    <select class="form-control" id="data-type" >
                        <option value="BF16 / FP16" selected> BF16 / FP16 </option>
                    </select>
                </div>
            </div>


             <div class="form-group row">
                <label for="batch-size" title="Enter the Batch Size" class="col-sm-3 col-form-label fw-bold text-end"> Batch Size
                 <!-- <i class="fa fa-info-circle" aria-hidden="true"> </i> -->
                
                 </label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="batch-size" placeholder="" >
                </div>
            </div>

       

            <div class="form-group row">
                <label for="max-sequence-length" class="col-sm-3 col-form-label fw-bold text-end"> Max Sequence Length</label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="max-sequence-length" placeholder="" >
                </div>
            </div>


            <div class="form-group row">
                <label for="num-embeddings" class="col-sm-3 col-form-label fw-bold text-end"> Embedding Dimension</label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="num-embeddings" placeholder="" >
                </div>
            </div>



            <div class="form-group row">
                <label for="num-layers" class="col-sm-3 col-form-label fw-bold text-end"> Number of Layers</label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="num-layers" placeholder="" >
                </div>
            </div>
    
            
            <div id="warningMessage" class="alert alert-warning text-danger" style="display:none;"> Please check and enter valid values in model configuration. </div>


            <div id="submit-button-row" class="form-group row">
                <div class="col-sm-9 offset-sm-3 text-center">
                   <div class="mt-3">
                        <button  id="submit-button" type="submit" class="btn btn-primary ml-25"> Submit</button>
                    </div>
                </div>
            </div>        



        </div>
    
    </form>

        <div id="batch-size-form"  style="display:none;"> 

        <h2> Form 2 </h2>

        <div class="form-group">

            <label for="text-input2"> Text Input</label>
            <input type="text" class="form-control" id="text-input2" placeholder="Enter Text">
        </div>

        <div class="form-group">
            <label for="select-input2"> Select Input </label>
            <select class="form-control" id="select-input2" >
                <option value="val11" > Value 11</option>
                <option value="val12"> Value 12</option>
            </select>
        </div>

        <button type="submit" class="btn btn-primary"> Submit</button>


        </h2>
        </div>

        <div id="calculator-result" style="margin-bottom:50px;"> </div>



        <div id="reset-button-row" class="form-group row" style="display:none;margin-bottom:50px;">
            <div class="col-sm-9 offset-sm-3 text-center">
                <div class="mt-3">
                    <button  id="reset-button"  class="btn btn-primary ml-25"> Reset Calculator</button>
                </div>
            </div>
        </div>  


.. raw:: html


    <script>


        $(document).ready(function() {
            $('#formSelect').on('change',function() {
                var form=$(this).val();
                if(form=='compute-core-form'){
                    $('#compute-core-form').show();
                   // $('#batch-size-form').hide();
                }
                //else if(form=='batch-size-form'){
                //    $('#batch-size-form').show();
                //    $('#compute-core-form').hide();
                //}
            });
            

            $('#model-select').on('change',function() {
                var modelSelected=$(this).val();
                if(modelSelected=='opt-66b'){
                   $("#batch-size").val("16");
                   $("#max-sequence-length").val("2048");
                   $("#num-embeddings").val("9216");
                   $("#num-layers").val("64");

                }
                else if(modelSelected=='custom-llm-model')
                {
                   $("#batch-size").val("");
                   $("#max-sequence-length").val("");
                   $("#num-embeddings").val("");
                   $("#num-layers").val("");

                }
              
            });

     $('#compute-core-form').show();
            $('#batch-size-form').hide();

        });

        
                function submitComputeCoreForm() {


                    require(['mathjs'], function(math) {

                    const batchSize = math.bignumber(parseInt($("#batch-size").val()));

                    const maxSequenceLength = math.bignumber(parseInt($("#max-sequence-length").val()));
                    
                    const numEmbeddings = math.bignumber(parseInt($("#num-embeddings").val()));
                    const numLayers = math.bignumber(parseInt($("#num-layers").val()));

                    const dTypeSize = math.bignumber(2);
                    

                    const weightMemFootPrintBytes = math.multiply(12,numLayers,math.pow(numEmbeddings,2),dTypeSize);
                    const weightMemFootPrintGB = math.divide(weightMemFootPrintBytes,math.pow(1024,3))


                    const kvCacheMemFootPrintBytes = math.multiply(batchSize,numLayers,maxSequenceLength,numEmbeddings,2,dTypeSize);
                    const kvCacheMemFootPrintGB = math.divide(kvCacheMemFootPrintBytes,math.pow(1024,3))

                    const memFootPrintGB = math.add(weightMemFootPrintGB,kvCacheMemFootPrintGB);

                    const numCoresCeiled = math.ceil(math.divide(memFootPrintGB,16));

                    const dataTypeSelected= $("#data-type").val();

                    const modelSelected= $("#model-select").val();
                    const instanceTypeSelected = $("#instance-type").val();


                    var warningMessage = document.getElementById('warningMessage')

                    if(isNaN(batchSize) || isNaN(numEmbeddings) || isNaN(maxSequenceLength) || isNaN(numLayers) || batchSize<=0 || numEmbeddings<=0 || maxSequenceLength<=0 || numLayers<=0 )
                    {
                        event.preventDefault();
                        warningMessage.style.display = 'block';
                        return false;
                    }
                    else
                    {
                        warningMessage.style.display = 'none';

                    }



                    var neuronCoresNeeded = -1

                    var tensorParallelDegreesSupported = [];
                    if (instanceTypeSelected == 'Trn1') {
                        tensorParallelDegreesSupported = [2,8,32];
                    }
                    else if (instanceTypeSelected == 'Inf2') {
                        tensorParallelDegreesSupported = [2,4,8,12,24];
                    }


                    for (let i=0; i < tensorParallelDegreesSupported.length; i++) {
                        if(numCoresCeiled <= tensorParallelDegreesSupported[i]) {
                            neuronCoresNeeded = tensorParallelDegreesSupported[i];
                            break;
                        }
                    }

                    // $('#calculator-result').html('<br><br><h3> Number of Neuron Cores needed: ' + neuronCoresNeeded + " </h3> ");

                    $('#submit-button-row').hide();
                    $('#neuron-calculator-select').hide();


                    if(neuronCoresNeeded>0)
                    {
                        $('#calculator-result').replaceWith('<div id="calculator-result" style="text-align:center;margin-bottom:50px;" > <b> Number of Neuron Cores needed: ' +  '<span style="font-size:22px;">' + neuronCoresNeeded + '</b></span></div>' );
                    }
                    else if(batchSize>1)
                    {
                        $('#calculator-result').replaceWith('<div id="calculator-result" style="text-align:center;margin-bottom:50px;" > <b> The model does not fit in a single instance. Multiple instances are needed to hold this model (calculator to be updated soon). Alternatively, consider reducing the batch size. </b></span></div>' );

                    }
                    else{
                       $('#calculator-result').replaceWith('<div id="calculator-result" style="text-align:center;margin-bottom:50px;" > <b> The model does not fit in a single instance. Multiple instances are needed to hold this model (calculator to be updated soon). </b></span></div>' );

                    }

                    $('#reset-button-row').show();


                    //css('background-color', '#f1f1f1')
                    $('#model-select').replaceWith('<span id="model-select" class="readonly-text" style="margin-top:5px;display:flex;">' + modelSelected + '</span>');
                    $('#instance-type').replaceWith('<span id="instance-type" class="readonly-text" style="margin-top:5px;display:flex;" >' + instanceTypeSelected + '</span>');
                     $('#data-type').replaceWith('<span id="data-type" class="readonly-text" style="margin-top:5px;display:flex;" >' + dataTypeSelected + '</span>');
                    $('#batch-size').replaceWith('<span id="batch-size" class="readonly-text" style="margin-top:5px;display:flex;">' + batchSize + '</span>');
                    $('#max-sequence-length').replaceWith('<span id="max-sequence-length" class="readonly-text" style="margin-top:5px;display:flex;">' + maxSequenceLength + '</span>');
                    $('#num-embeddings').replaceWith('<span id="num-embeddings" class="readonly-text" style="margin-top:5px;display:flex;">' + numEmbeddings + '</span>');
                     $('#num-layers').replaceWith('<span id="num-layers" class="readonly-text" style="margin-top:5px;display:flex;">' + numLayers + '</span>');





                    });

                    return false;
                
    }



        
  

            $('#form1').on('submit',function(e)
            {
                e.preventDefault();
                var text2 = $('#text1').val();
                var select2 = $('#select2').val();
                $('#form2-result').html('User selected' + select1 + 'from dropdown');
            });


         function resetNeuronCalculator() {

            location.reload();


         }


          document.getElementById("reset-button").addEventListener("click",function() { resetNeuronCalculator(); } );


        $(function() {
            $('[data-toggle="tooltip"]').tooltip();
            }
        );

    </script>

