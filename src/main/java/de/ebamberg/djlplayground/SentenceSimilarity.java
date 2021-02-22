/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package de.ebamberg.djlplayground;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author erik.bamberg@web.de
 *
 */
public class SentenceSimilarity {


    private static final String[] internetExample = new String [] {
    	"How can I increase the speed of my internet connection while using a VPN?",
    	"How can i increase speed of internet ?",
    	"Why is my internet so slow?",
    	"Where is London?"
    };
    
    private static final String[] newspaperExample = new String [] {
	    	"Obama speaks to the media in Illinois",
	    	"The President greets the press in Chicago",
	    	"The President speaks to the newspapers in Illinois",
	    	"The President speaks to the media in Washington",
	    	"Obama greets the press in Illinois",
	    	"The Queen of England writes a letter to Obama",
	    	"Boris Johnson wrote on twitter."
	    };
    
    /**
     * main.
     * @param args
     * @throws IOException 
     * @throws MalformedModelException 
     * @throws ModelNotFoundException 
     */
    public static void main(String[] args) throws Exception {
        String[] inputs = internetExample;
	
	
        String modelUrl =
                "https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz";

        Criteria<String[], List> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.TEXT_EMBEDDING)
                        .setTypes(String[].class, List.class)
                        .optModelUrls(modelUrl)
                        .optTranslator(new CosineSimilarityTranslator())
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<String[], List> model = ModelZoo.loadModel(criteria);
                Predictor<String[], List> predictor = model.newPredictor()) {
            List result=predictor.predict(inputs);
            System.out.println("similarities to sentence:"+inputs[0]);
            IntStream.range(1, inputs.length).forEach( i -> {
        	System.out.println(result.get(i)+"\t=>\t"+"\t"+inputs[i]);
            });
        }

    }
    
    private static final class CosineSimilarityTranslator implements Translator<String[], List> {

        @Override
        public NDList processInput(TranslatorContext ctx, String[] inputs) {
            // manually stack for faster batch inference
            NDManager manager = ctx.getNDManager();
            NDList inputsList =
                    new NDList(
                            Arrays.stream(inputs)
                                    .map(manager::create)
                                    .collect(Collectors.toList()));
            return new NDList(NDArrays.stack(inputsList));
        }

        @Override
        public List<Float> processOutput(TranslatorContext ctx, NDList list) {
            NDList result = new NDList();
            long numOutputs = list.singletonOrThrow().getShape().get(0);
            for (int i = 0; i < numOutputs; i++) {
                result.add(list.singletonOrThrow().get(i));
            }
            
            List<Float> similarities=new ArrayList<>(result.size());
           
            similarities.add(1f);
            for (int i=1; i<result.size(); i++) {
        	similarities.add(cosineSimilarity(result.get(0),result.get(i)));
            }
            
            return similarities;
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
        
        public static float cosineSimilarity(NDArray vectorA, NDArray vectorB) {
            // NDArray dotProduct = vectorA.dot(vectorB); // not implemented in tensorflow
            NDArray dotProduct = vectorA.mul(vectorB).sum();    
            NDArray normA = vectorA.pow(2).sum().sqrt();
            NDArray normB = vectorB.pow(2).sum().sqrt(); 
            NDArray cosineSim=dotProduct.div (normA.mul(normB));
            return cosineSim.getFloat();
            
        }
        
//        public static float cosineSimilarity(float[] vectorA, float[] vectorB) {
//            float dotProduct = 0.0f;
//            float normA = 0.0f;
//            float normB = 0.0f;
//            for (int i = 0; i < vectorA.length; i++) {
//                dotProduct += vectorA[i] * vectorB[i];
//                normA += Math.pow(vectorA[i], 2);
//                normB += Math.pow(vectorB[i], 2);
//            }   
//            return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
//        }
        
    }

}
