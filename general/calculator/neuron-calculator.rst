.. _neuron_calculator:

Neuron Calculator
=================

.. raw:: html

       <link href="https://cdnjs.cloudflare.com/ajax/libs/choices.js/1.1.6/styles/css/choices.min.css" rel="stylesheet">

    <script src="https://cdnjs.cloudflare.com/ajax/libs/choices.js/1.1.6/choices.min.js"></script>


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
                        <option value="compute-core-form"> NeuronCores needed for LLM Inference</option>
                    </select>

                </div>
            </div>


       <form id="compute-core-form" method="get" onkeydown="return event.key != 'Enter';"  onsubmit="submitComputeCoreForm(); return false;"> 

            <h2> Number of NeuronCores needed for LLM Inference</h2>
            <span style="margin-bottom:5px;">Please enter model configurations (You can enter multiple values of each hyperparameter. Press enter after adding each value in the text field) </span>

            
                 <div class="form-group row">
                    <label for="model-select" class="col-sm-3 col-form-label fw-bold text-end"> Model: </label>
                   <div class="col-sm-8">

                        <select class="form-control" id="model-select" style="width:100%;">
                            <option value="custom-llm-model"> Custom LLM Model </option>

                            <optgroup label="Sample Model Configuration" class="font-weight-bold" >
                                <option value="opt-66b"> opt-66b </option>
                                <option value="meta-llama/llama-2-7b"> meta-llama/Llama-2-7b </option>
                                <option value="meta-llama/llama-2-13b"> meta-llama/Llama-2-13b </option>
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
                <label for="batch-size" title="Enter the Batch Size" class="col-sm-3 col-form-label fw-bold text-end"> Batch Size:
                 <!-- <i class="fa fa-info-circle" aria-hidden="true"> </i> -->
                 </label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="batch-size" placeholder="" >        
                </div>
            </div>

       

            <div class="form-group row">
                <label for="max-sequence-length" class="col-sm-3 col-form-label fw-bold text-end"> Max Sequence Length:</label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="max-sequence-length" placeholder="" >
                </div>
            </div>


            <div class="form-group row">
                <label for="num-embeddings" class="col-sm-3 col-form-label fw-bold text-end"> Embedding Dimension:</label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="num-embeddings" placeholder="" >
                </div>
            </div>

            <div class="form-group row">
                <label for="num-attention-heads" class="col-sm-3 col-form-label fw-bold text-end"> Number of Attention Heads:</label>
                <div class="col-sm-8">
                    <input type="text" class="form-control" id="num-attention-heads" placeholder="" >
                </div>
            </div>


            <div class="form-group row">
                <label for="num-layers" class="col-sm-3 col-form-label fw-bold text-end"> Number of Layers:</label>
                <div class="col-sm-8">
                    <input type="text" name="num-layers" class="form-control" id="num-layers" >              
                </div>
            </div>
    
        
            <div class="form-group row">
                <div class="form-check">
                    <input type="checkbox" style="width:20px;height:20px;margin-left:5px;" name="num-attention-heads-divisible" class="form-check-input" id="num-attention-heads-divisible" >              
                    <label for="num-attention-heads-divisible" class="form-check-label" style="margin-left:35px;"> Tensor Parallel Degree Constraint (Flexible tensor parallelism (TP) is not supported for certain models like GPT-J and GPT-NeoX in transformers-neuronx. Checking this box will flag a TP degree as invalid if the number of attention heads is not divisible by it.) </label>
                </div>
            </div>



            <div id="warningMessage" class="alert alert-warning text-danger" style="display:none;"> Invalid model configurations entered. Each text field accepts multiple values. Please press Enter after adding a new value to the text field.</div>


            <div id="submit-button-row" class="form-group row">
                <div class="col-sm-9 offset-sm-3 text-center">
                   <div class="mt-3">
                        <button  id="submit-button" type="submit" class="btn btn-primary ml-25"> Submit</button>
                    </div>
                </div>
            </div>        



        </div>
    
    </form>

      
        <div id="calculator-result" style="margin-bottom:50px;"> </div>

        <table id="results-table" class="table" style="display:none;">
            <thead>
                <tr>
                    <th> Batch Size </th>
                    <th> Max Seq Length </th>
                    <th> Embedding Dimension </th>
                    <th> Num Attention Heads </th>
                    <th> Num Layers </th>
                    <th> Memory Footprint (GB)</th>
                    <th> TP Degree(NeuronCores) </th>
                    <th> Instances Recommended </th>
                </tr>
            </thead>
            <tbody id="results-body">
            </tbody>
       </table>


     

        <div id="reset-button-row" class="form-group row" style="display:none;margin-bottom:50px;">
            
               <div class="col-sm-9 offset-sm-2 text-center">
                <div class="mt-3">
                    <button  id="edit-button"  class="btn btn-primary ml-15 mr-3"> Edit Model Configuration</button>  
                    <button  id="reset-button"  class="btn btn-primary ml-25"> Reset Calculator</button>  

                    <!--<button  id="reset-button"  class="btn btn-primary ml-25"> Reset Calculator</button> -->

                </div>
             </div>
            
        </div>  


        <style>
           .choices__list--multiple .choices__item {
                background-color: #6c757d;
                color: #ffffff;
           }
          
          .table {
                border-top: 1px solid black;
                margin-top: 20px;
                margin-bottom: 20px;
          } 
          .green-row {
             background-color: #c8e6c9 ;
             }

           .table-row {
              border-bottom: 1px solid black;
              } 


           </style>

.. raw:: html

    
    <script>

        var numLayersField ;
        var batchSizeField;
        var maxSequenceLengthField;
        var numEmbeddingDimensionField;
        var numAttentionHeadsField;


        var modelSelectSavedHTML = "";
        var modelSelectSavedValue = "";
        var instanceTypeSavedHTML = "";
        var instanceTypeSavedValue = "";
        var dataTypeSavedHTML = "";
        var dataTypeSavedValue = "";


        document.addEventListener('DOMContentLoaded', function() {

            numLayersField = new Choices('#num-layers' , { 
               maxItemCount: 8,
               valueField: 'id',
               labelField: 'title',
               searchField: 'title',
               shouldSort: false ,
               searchEnabled: false ,
               create: true ,
               removeItemButton:true,
               duplicateItems: false,
            });

            batchSizeField = new Choices('#batch-size' , { 
               maxItemCount: 8,
               valueField: 'id',
               labelField: 'title',
               searchField: 'title',
               shouldSort: false ,
               searchEnabled: false ,
               create: true ,
               removeItemButton:true,
               duplicateItems: false,
            });

            maxSequenceLengthField = new Choices('#max-sequence-length' , { 
               maxItemCount: 8,
               valueField: 'id',
               labelField: 'title',
               searchField: 'title',
               shouldSort: false ,
               searchEnabled: false ,
               create: true ,
               removeItemButton:true,
               duplicateItems: false,
            });

            numEmbeddingDimensionField = new Choices('#num-embeddings' , { 
               maxItemCount: 8,
               valueField: 'id',
               labelField: 'title',
               searchField: 'title',
               shouldSort: false ,
               searchEnabled: false ,
               create: true ,
               removeItemButton:true,
               duplicateItems: false,
            });


            numAttentionHeadsField = new Choices('#num-attention-heads' , { 
               maxItemCount: 8,
               valueField: 'id',
               labelField: 'title',
               searchField: 'title',
               shouldSort: false ,
               searchEnabled: false ,
               create: true ,
               removeItemButton:true,
               duplicateItems: false,
            });

        });


        function modelSelectOnChangeHandler() {
             var modelSelected=$(this).val();
                if(modelSelected=='opt-66b'){

                    batchSizeField.clearStore();
                    batchSizeField.setValue([{value: '16', label: '16'},]);

                    maxSequenceLengthField.clearStore();
                    maxSequenceLengthField.setValue([{value: '2048', label: '2048'},]);

                    
                    numEmbeddingDimensionField.clearStore();
                    numEmbeddingDimensionField.setValue([{value: '9216', label: '9216'},]);


                    numLayersField.clearStore();
                    numLayersField.setValue([{value: '64', label: '64'},]);


                    numAttentionHeadsField.clearStore();
                    numAttentionHeadsField.setValue([{value: '72', label: '72'},]);

                }
                else if(modelSelected == 'meta-llama/llama-2-7b'){

                    batchSizeField.clearStore();
                    batchSizeField.setValue([{value: '16', label: '16'},]);

                    maxSequenceLengthField.clearStore();
                    maxSequenceLengthField.setValue([{value: '4096', label: '4096'},]);

                    
                    numEmbeddingDimensionField.clearStore();
                    numEmbeddingDimensionField.setValue([{value: '4096', label: '4096'},]);


                    numLayersField.clearStore();
                    numLayersField.setValue([{value: '32', label: '32'},]);


                    numAttentionHeadsField.clearStore();
                    numAttentionHeadsField.setValue([{value: '32', label: '32'},]);

                }
                else if(modelSelected == 'meta-llama/llama-2-13b'){

                    batchSizeField.clearStore();
                    batchSizeField.setValue([{value: '16', label: '16'},]);

                    maxSequenceLengthField.clearStore();
                    maxSequenceLengthField.setValue([{value: '4096', label: '4096'},]);

                    
                    numEmbeddingDimensionField.clearStore();
                    numEmbeddingDimensionField.setValue([{value: '5120', label: '5120'},]);


                    numLayersField.clearStore();
                    numLayersField.setValue([{value: '40', label: '40'},]);


                    numAttentionHeadsField.clearStore();
                    numAttentionHeadsField.setValue([{value: '40', label: '40'},]);

                }
                else if(modelSelected=='custom-llm-model')
                {
                    batchSizeField.clearStore();
                    maxSequenceLengthField.clearStore();
                    numEmbeddingDimensionField.clearStore();
                    numLayersField.clearStore();
                    numAttentionHeadsField.clearStore();


                }
                else if(modelSelected=='import-hf-model')
                {
                    var hfDivLabel= document.getElementById('hf-model-url-label');
                    hfDivLabel.style.display = 'block';

                    var hfDiv= document.getElementById('hf-model-url-input-field');
                    hfDiv.style.display = 'block';

                    var hfDivButton= document.getElementById('hf-model-url-import-button');
                    hfDivButton.style.display = 'block';
                  
                    document.getElementById("hf-model-url-import-button").addEventListener("click",function() { processHFImport(); } );

                }


        }


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
            

            $('#model-select').on('change',modelSelectOnChangeHandler);





     $('#compute-core-form').show();
            $('#batch-size-form').hide();

        });
                function processHFImport() {
                        var hfModelURL= $("#hf-model-url").val();
                        var hfModelJSONURL = hfModelURL + "/raw/main/config.json";

                        var xhr = new XMLHttpRequest();
                        xhr.open('GET',hfModelJSONURL,true);

                        xhr.onload = function() {
                          if(xhr.status==200)
                          {
                                var data = JSON.parse(xhr.responseText);

                                var numLayersVal = -1;
                                var numEmbeddingsVal = -1;

                                if ('n_layer' in data)
                                {
                                    numLayersVal = data.n_layer;
                                }


                                if ('hidden_size' in data)
                                {
                                    numEmbeddingsVal = data.hidden_size;
                                }


                                if(numLayersVal > -1)
                                {
                                    numLayersField.clearStore();
                                    numLayersField.setValue([{value: numLayersVal, label: numLayersVal},]);
                                }


                                if(numEmbeddingsVal > -1)
                                {
                                    numEmbeddingDimensionField.clearStore();
                                    numEmbeddingDimensionField.setValue([{value: numEmbeddingsVal, label: numEmbeddingsVal},]);
                                }



                          }
                        };

                        xhr.send();    

                }
        
                function submitComputeCoreForm() {


                    require(['mathjs'], function(math) {

                 


                    batchSizeVals = batchSizeField.getValue(true);
                    maxSequenceLengthVals = maxSequenceLengthField.getValue(true);
                    numEmbeddingDimensionVals = numEmbeddingDimensionField.getValue(true);
                    numAttentionHeadsVals = numAttentionHeadsField.getValue(true);


                    var numAttentionHeadsDivisibleField = document.getElementById("num-attention-heads-divisible");
                    attentionHeadsConstraint = false;

                    if(numAttentionHeadsDivisibleField.checked)
                    {
                        attentionHeadsConstraint = true;
                    }


                    numLayersVals = numLayersField.getValue(true);

                    const dataTypeSelected= $("#data-type").val();
                    const dTypeSize = math.bignumber(2);
                    
                    const instanceTypeSelected = $("#instance-type").val();

                    const modelSelected= $("#model-select").val();

                    //var hfModelURL= $("#hf-model-url").val();

                    var calculatorResultStr = '';

                    var resultsTable = document.getElementById('results-table');
                    var resultsBody = document.getElementById('results-body');
                    var warningMessage = document.getElementById('warningMessage')

                
                   var inf2Cores = {'Inf2.xlarge':2 , 'Inf2.8xlarge':2 , 'Inf2.24xlarge':12 , 'Inf2.48xlarge':24 };
                   var inf2Keys = Object.keys(inf2Cores);

                   var trn1Cores = { 'Trn1.2xlarge':2 , 'Trn1.32xlarge':32};
                   var trn1Keys = Object.keys(trn1Cores);
             


                    if(batchSizeVals=== null || numEmbeddingDimensionVals=== null || maxSequenceLengthVals=== null || numLayersVals=== null || (batchSizeVals.length === 0) || (maxSequenceLengthVals.length === 0) || (numEmbeddingDimensionVals.length === 0) || (numAttentionHeadsVals.length === 0) || (numLayersVals.length  === 0) )
                    {
                         event.preventDefault();
                         warningMessage.style.display = 'block';
                         return false;
                    }


                    rowBackgroundColor = "#f5f5f5" ;

                    for(let i=0; i<batchSizeVals.length;  i++) {
                        for(let j=0; j<maxSequenceLengthVals.length;  j++) {
                            for(let k=0; k<numEmbeddingDimensionVals.length;  k++) {
                                for(let m=0; m<numAttentionHeadsVals.length;  m++) {
                                    for(let l=0; l<numLayersVals.length;  l++) {
                                        
                                           
                                          rowBackgroundColor = (rowBackgroundColor === "#f5f5f5") ? "#e0e0e0" : "#f5f5f5"
                                        

                                           batchSize = math.bignumber(parseInt(batchSizeVals[i]));
                                           maxSequenceLength = math.bignumber(parseInt(maxSequenceLengthVals[j]));
                                           numEmbeddings = math.bignumber(parseInt(numEmbeddingDimensionVals[k]));
                                           numLayers = math.bignumber(parseInt(numLayersVals[l]));
                                           
                                           numAttentionHeads =  math.bignumber(parseInt(numAttentionHeadsVals[m]));


                                           weightMemFootPrintBytes = math.multiply(12,numLayers,math.pow(numEmbeddings,2),dTypeSize);
                                           weightMemFootPrintGB = math.divide(weightMemFootPrintBytes,math.pow(1024,3))


                                           kvCacheMemFootPrintBytes = math.multiply(batchSize,numLayers,maxSequenceLength,numEmbeddings,2,dTypeSize);
                                           kvCacheMemFootPrintGB = math.divide(kvCacheMemFootPrintBytes,math.pow(1024,3))

                                          
                                           memFootPrintGB = math.add(weightMemFootPrintGB,kvCacheMemFootPrintGB);
                                           memFootPrintGBRounded = math.ceil(memFootPrintGB)

                                           numCoresCeiled = math.ceil(math.divide(memFootPrintGB,16));


                                            if(isNaN(batchSize) || isNaN(numEmbeddings) || isNaN(maxSequenceLength) || isNaN(numLayers) || batchSize<=0 || numEmbeddings<=0 || numAttentionHeads<=0 || maxSequenceLength<=0 || numLayers<=0 )
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


                                            //alert("tensor parallel degrees supported on instance:" + tensorParallelDegreesSupported);

                                            //alert("num cores ceiled:" + numCoresCeiled)


                                            var tpDegreesPossible = [];

                                            for (let p=0; p < tensorParallelDegreesSupported.length; p++) {
                                                if(numCoresCeiled <= tensorParallelDegreesSupported[p]) {
                                                    neuronCoresNeeded = tensorParallelDegreesSupported[p];
                                                    
                                                    for(let q=p ; q < tensorParallelDegreesSupported.length; q++) {
                                                          var curPossibleTPDegree = tensorParallelDegreesSupported[q]
                                                          tpDegreesPossible.push(curPossibleTPDegree);
                                                    }

                                                    break;
                                                }
                                            }


                                            //alert("tp degrees possible:" + tpDegreesPossible)

                                            if(tpDegreesPossible.length == 0)
                                            {

                                                var row = document.createElement('tr');
                                                row.style.backgroundColor = rowBackgroundColor

                                                var batchSizeCell = document.createElement('td');
                                                var maxSequenceLengthCell = document.createElement('td');
                                                var numEmbeddingsCell = document.createElement('td');
                                                var numAttentionHeadsCell = document.createElement('td');
                                                var numLayersCell = document.createElement('td');
                                                var memoryFootprintCell = document.createElement('td');
                                                var numCoresCell = document.createElement('td');
                                                var instancesSupportedCell = document.createElement('td');


                                                batchSizeCell.textContent =  batchSize;
                                                maxSequenceLengthCell.textContent =  maxSequenceLength;
                                                numEmbeddingsCell.textContent =  numEmbeddings;
                                                numAttentionHeadsCell.textContent = numAttentionHeads;
                                                numLayersCell.textContent =  numLayers;

                                                memoryFootprintCell.textContent = memFootPrintGBRounded


                                                numCoresCell.textContent = "N/A";

                                                instancesSupportedCell.style.color = 'red' ;

                                                instancesSupportedCellStr = "Does not fit in Single Instance. Multiple Instances needed"


                                                var rawHtmlElement = document.createElement('div');
                                                rawHtmlElement.innerHTML = instancesSupportedCellStr
                                                var sphinxHTMLString = '\n\n    ' + rawHtmlElement.outerHTML;
                                                instancesSupportedCell.innerHTML = sphinxHTMLString

                                                row.classList.add('table-row');


                                                row.appendChild(batchSizeCell);
                                                row.appendChild(maxSequenceLengthCell);
                                                row.appendChild(numEmbeddingsCell);
                                                row.appendChild(numAttentionHeadsCell);
                                                row.appendChild(numLayersCell);
                                                row.appendChild(memoryFootprintCell);
                                                row.appendChild(numCoresCell);
                                                row.appendChild(instancesSupportedCell);

                                                //alert(row.innerHTML)

                                                resultsBody.appendChild(row);



                                            }
                                            
                                            
                                    
                                            for (let q=0;q<tpDegreesPossible.length; q++)
                                            {
                                                var row = document.createElement('tr');

                                                row.style.backgroundColor = rowBackgroundColor

                                                var batchSizeCell = document.createElement('td');
                                                var maxSequenceLengthCell = document.createElement('td');
                                                var numEmbeddingsCell = document.createElement('td');
                                                var numAttentionHeadsCell = document.createElement('td');
                                                var numLayersCell = document.createElement('td');
                                                var memoryFootprintCell = document.createElement('td');
                                                var numCoresCell = document.createElement('td');
                                                var instancesSupportedCell = document.createElement('td');

                                                batchSizeCell.textContent =  batchSize;
                                                maxSequenceLengthCell.textContent =  maxSequenceLength;
                                                numEmbeddingsCell.textContent =  numEmbeddings;
                                                numAttentionHeadsCell.textContent = numAttentionHeads;
                                                numLayersCell.textContent =  numLayers;

                                                memoryFootprintCell.textContent = memFootPrintGBRounded

                                                var instancesSupportedCellStr = "";

                                                var tpDegree = tpDegreesPossible[q]
                                                if(neuronCoresNeeded>0 && (!attentionHeadsConstraint || (numAttentionHeads % tpDegree === 0)))
                                                {
                                                    numCoresCell.textContent = tpDegree

                                                    if(instanceTypeSelected === "Inf2")
                                                    {
                                                    for(var p=0; p<inf2Keys.length ; p++) {
                                                    var inf2InstanceSize = inf2Keys[p];
                                                    var instanceCores = inf2Cores[inf2InstanceSize]
                                                    if(instanceCores>=tpDegree)
                                                    {
                                                            if(instancesSupportedCellStr.length >0)  instancesSupportedCellStr += "<br>";
                                                            instancesSupportedCellStr += inf2InstanceSize
                                                    } 
                                                    }
                                                    }
                                                    else if(instanceTypeSelected === "Trn1")
                                                    {
                                                    for(var p=0; p<trn1Keys.length ; p++) {
                                                            var trn1InstanceSize = trn1Keys[p];
                                                            var instanceCores = trn1Cores[trn1InstanceSize]
                                                            if(instanceCores>=tpDegree)
                                                            {
                                                                    if(instancesSupportedCellStr.length >0)  instancesSupportedCellStr += "<br>";
                                                                    instancesSupportedCellStr += trn1InstanceSize
                                                            } 
                                                        }
                                                    } 


                                                    
                                                    if(instancesSupportedCellStr.length>0)
                                                    {
                                                        if(tpDegreesPossible.length>1)
                                                        {
                                                            if(q === 0)
                                                            {
                                                                numCoresCell.textContent = numCoresCell.textContent + "(Min NeuronCores Reqd)";
                                                            }
                                                            else if(q === (tpDegreesPossible.length-1))
                                                            {
                                                                numCoresCell.textContent = numCoresCell.textContent + "(Best Latency)";
                                                            }
                                                            else
                                                            {
                                                                numCoresCell.textContent = numCoresCell.textContent

                                                            }
                                                        }
                                                        else
                                                        {
                                                            numCoresCell.textContent = numCoresCell.textContent + "(Min NeuronCores Reqd)";

                                                        }
                                                    }
                                                    else
                                                    {

                                                        numCoresCell.textContent = numCoresCell.textContent;

                                                    }

                                                    instancesSupportedCell.textContent = "Inf2 or Trn1 instances"
                                                // row.classList.add('green-row')
                                                }
                                                else if((neuronCoresNeeded>0 && attentionHeadsConstraint && (numAttentionHeads % tpDegree != 0)))
                                                {

                                                    numCoresCell.textContent = tpDegree

                                                    instancesSupportedCell.style.color = 'red' ;


                                                    instancesSupportedCellStr = "TP degree not supported. Number of attention heads must be divisible by TP degree."
                                                    // row.classList.add('red-row')


                                                }
                                                else{

                                                    numCoresCell.textContent = "N/A";

                                                    instancesSupportedCell.style.color = 'red' ;


                                                    instancesSupportedCellStr = "Does not fit in Single Instance. Multiple Instances needed"
                                                    // row.classList.add('red-row')

                                                }


                                                var rawHtmlElement = document.createElement('div');
                                                rawHtmlElement.innerHTML = instancesSupportedCellStr
                                                var sphinxHTMLString = '\n\n    ' + rawHtmlElement.outerHTML;
                                                instancesSupportedCell.innerHTML = sphinxHTMLString

                                                row.classList.add('table-row');


                                                row.appendChild(batchSizeCell);
                                                row.appendChild(maxSequenceLengthCell);
                                                row.appendChild(numEmbeddingsCell);
                                                row.appendChild(numAttentionHeadsCell);
                                                row.appendChild(numLayersCell);
                                                row.appendChild(memoryFootprintCell);
                                                row.appendChild(numCoresCell);
                                                row.appendChild(instancesSupportedCell);

                                                //alert(row.innerHTML)

                                                resultsBody.appendChild(row);
                                            }

                                        }

                                    }
                                }
                            }
                        }



                    $('#submit-button-row').hide();
                    $('#neuron-calculator-select').hide();

                    var calculatorResult = document.getElementById('calculator-result');
                    calculatorResult.innerHTML = "For more details on how the number of NeuronCores is computed, please refer to the <a href='https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/transformers-neuronx/generative-llm-inference-with-neuron.html#how-many-neuroncores-do-i-need'>LLM Inference App Note</a>";
                    //$('#calculator-result').replaceWith("For more details on how the number of min NeuronCores are computed, please refer to the <a href='https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/transformers-neuronx/generative-llm-inference-with-neuron.html#how-many-neuroncores-do-i-need'>LLM Inference App Note</a>");

                    
                    resultsTable.style.display = 'block';
                     
          

                    $('#reset-button-row').show();


                    batchSizeField.disable();
                    maxSequenceLengthField.disable();
                    numEmbeddingDimensionField.disable();
                    numLayersField.disable();
                    numAttentionHeadsField.disable();

                    var numAttentionHeadsDivisibleField = document.getElementById("num-attention-heads-divisible");
                    numAttentionHeadsDivisibleField.disabled = true;

                    const choiceCloseButtons = document.querySelectorAll('.choices__button');
                    choiceCloseButtons.forEach(button => { button.disabled=true;} );



                    modelSelectSavedHTML =  document.getElementById("model-select").outerHTML;
                    modelSelectSavedValue = modelSelected
                    instanceTypeSavedHTML =  document.getElementById("instance-type").outerHTML;
                    instanceTypeSavedValue = instanceTypeSelected
                    dataTypeSavedHTML =  document.getElementById("data-type").outerHTML;
                    dataTypeSavedValue = dataTypeSelected

                   // $('#hf-model-url').replaceWith('<span id="hf-model-url" class="readonly-text" style="margin-top:5px;display:flex;">' + hfModelURL + '</span>');
                    $('#model-select').replaceWith('<span id="model-select" class="readonly-text" style="margin-top:5px;display:flex;">' + modelSelected + '</span>');
                    $('#instance-type').replaceWith('<span id="instance-type" class="readonly-text" style="margin-top:5px;display:flex;" >' + instanceTypeSelected + '</span>');
                     $('#data-type').replaceWith('<span id="data-type" class="readonly-text" style="margin-top:5px;display:flex;" >' + dataTypeSelected + '</span>');
                  

                  


                    });

                    return false;
                
    }



        

         function editNeuronCalculatorConfiguration() {
                    batchSizeField.enable();
                    maxSequenceLengthField.enable();
                    numEmbeddingDimensionField.enable();
                    numLayersField.enable();
                    numAttentionHeadsField.enable();

                    var numAttentionHeadsDivisibleField = document.getElementById("num-attention-heads-divisible");
                    numAttentionHeadsDivisibleField.disabled = false;

                    const choiceCloseButtons = document.querySelectorAll('.choices__button');
                    choiceCloseButtons.forEach(button => { button.disabled=false;} );


                    $('#reset-button-row').hide();

                    $('#submit-button-row').show();
                    $('#neuron-calculator-select').show();


                    document.getElementById('model-select').outerHTML = modelSelectSavedHTML;
                    document.getElementById('model-select').value = modelSelectSavedValue;

            
                    document.getElementById('model-select').addEventListener("change",modelSelectOnChangeHandler)



                    document.getElementById('instance-type').outerHTML = instanceTypeSavedHTML;
                    document.getElementById('instance-type').value = instanceTypeSavedValue;

                    document.getElementById('data-type').outerHTML = dataTypeSavedHTML;
                    document.getElementById('data-type').value = dataTypeSavedValue;

                    var calculatorResult = document.getElementById('calculator-result');
                    calculatorResult.innerHTML = "";

                    var resultsTable = document.getElementById('results-table');

                    for (var i=resultsTable.rows.length-1; i>0; i--) 
                    {
                        resultsTable.deleteRow(i);
                    }
                    resultsTable.style.display = 'none';
                 


           
         }


         function resetNeuronCalculator() {

            location.reload();


         }

          document.getElementById("edit-button").addEventListener("click",function() { editNeuronCalculatorConfiguration(); } );
          document.getElementById("reset-button").addEventListener("click",function() { resetNeuronCalculator(); } );


        $(function() {
            $('[data-toggle="tooltip"]').tooltip();
            }
        );

    </script>

