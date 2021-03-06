import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Vector;

/*
This file will create a dictionary from the training set.
The dictionary is a global dictionary and will also be used for the test set.

A dictionary of {word : id} pairs.

This object also convert the ".java" files into 'id' files, replacing words with their dictionary ids.

Author: K. Sai Srinivas
Project: SCAA
Date: 29th April, 2016.
*/

/*************************************************************************************************************************************/
//A class that creates a dictionary for the training set
//Creates new files that contain the IDs instead of words, for every doc in the training corpus.

public class LDAData {
	
	//------------------------------------------------------
	//The variables
	//------------------------------------------------------
	public Dictionary localDict;	//a dictionary object for this training corpur
	public int M;					//number of documents in the corpus
	public int V;					//number of words in the corpus
	
	public Document[] docs;			//an array of documents in the corpus
	
	int flag;						//train-test flag. 0 and 1.
	//------------------------------------------------------
	//The constructor
	//------------------------------------------------------
	/***************************/
	public LDAData(){
		localDict = new Dictionary();
		M = 0;
		V = 0;
		docs = null;
		//default TRAIN
		flag = 0;
	}
	
	/***************************/
	public LDAData(int M){
		localDict = new Dictionary();
		this.M = M;
		this.V = 0;
		docs = new Document[M];
	}
	
	//------------------------------------------------------
	//Document setter methods
	//------------------------------------------------------
	/***************************/
	//This method sets the document at the given index.
	public void setDocument(Document doc, int index){
		if(index >=0 && index < M){
			docs[index] = doc;
		}
	}
	
	/***************************/
	//Set the document at the given index from a string
	public void SetDocument(String str, int index){
		//variables
		String[] words;
		Vector<Integer> IDS = new Vector<Integer>();
		
		//check if index is correct
		if(index >= 0 && index < M){
			words = str.split("[ ]");
			
			//read the words
			for(int i=0; i<words.length; i++){
				//get the ID of the word
				if(flag == 0){
					int id = localDict.addWords(words[i]);
					IDS.add(id);
				}
				else{
					//Do not update the dict
					//Do not add word if not present in the train dictionary
					if(localDict.containsWord(words[i])){
						IDS.add(localDict.getID(words[i]));
					}
				}
			}
		}
		
		Document doc = new Document(IDS);
		docs[index] = doc;
		this.V = localDict.wordToId.size();
	}
	
	//------------------------------------------------------
	//I/O methods
	//------------------------------------------------------
	/***************************/
	//@parameters filename: File containing paths of all the documents
	//Assume existence of a file with M and paths of all the documents
	//A very important method
	public boolean createData(String filename) throws IOException{
		//Variables
		String line;
		
		BufferedReader rd = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
		
		//The first line is M
		int m = Integer.parseInt(rd.readLine());
		this.M = m;
		
		for(int i=0; i<m; i++){
			//read each line from the file and create a document for the file
			//TODO: set absolute pathnames
			line = rd.readLine();
			readFile(line, i);
		}
		
		rd.close();
		return true;
	}
	
	/***************************/
	//@parameters filename: File containing paths of all the documents
	//Assume existence of a file with M and paths of all the documents
	//A very important method
	public boolean createData(String filename, Dictionary localDict) throws IOException{
		//Variables
		String line;
		
		//set the flag as test
		flag = 1;
		
		BufferedReader rd = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
		
		//The first line is M
		int m = Integer.parseInt(rd.readLine());
		this.M = m;
		
		//set the dictionary
		this.localDict = localDict;
		
		for(int i=0; i<m; i++){
			//read each line from the file and create a document for the file
			//TODO: set absolute pathnames
			line = rd.readLine();
			readFile(line, i);
		}
		
		rd.close();
		return true;
	}
	
	/***************************/
	//Function to read data from file and fill documents
	public boolean readFile(String filename, int index) throws IOException{
		//variables
		String line;
		
		line = new String(Files.readAllBytes(Paths.get(filename)), StandardCharsets.UTF_8);
		
		//set the document at this index
		this.SetDocument(line, index);
		
		return true;
		
	}
	
}

/***********************************************************************************************************************************/
//A class that defines the dictionary for the corpus. Note that we use the training set as the dictionary
//corpus and construct the dictionary accordingly.

class Dictionary{
	
	//------------------------------------------------------
	//The variables
	//------------------------------------------------------
	public Map<String, Integer> wordToId;	//map from word to id for all encountered words in the training set
	public Map<Integer, String> idToWord;	//map from id to word for all encountered words in the training set
	
	//------------------------------------------------------
	//The constructor
	//------------------------------------------------------
	public Dictionary(){
		wordToId = new HashMap<String, Integer>();
		idToWord = new HashMap<Integer, String>();
	}

	//------------------------------------------------------
	//Creation Methods
	//------------------------------------------------------
	/***************************/
	//This function adds words to the dictionary and returns the id
	public int addWords(String word){
		//check if word is present in the dictionary
		if(containsWord(word) == true){
			return getID(word);
		}
		else{
			//The id is the size of the dictionary for ease
			int id = wordToId.size();
			wordToId.put(word, id);
			idToWord.put(id, word);
			
			return id;
		}	
	}
	
	//------------------------------------------------------
	//I/O Methods
	//------------------------------------------------------
	/***************************/
	//Read the dictionary from a word map file
	//filename contains word id pairs in every line
	public boolean readWordFile(String filename) throws IOException{
		//Variables
		String line;
		int nwords;
		
		BufferedReader rd = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
		
		//Reading the whole file in a shot
		line= rd.readLine();
		nwords = Integer.parseInt(line);
		
		//read
		for(int i=0; i < nwords; i++){
			line = rd.readLine();
			String[] strTok = line.split("[ ]");
			
			if(strTok.length != 2) continue;
			
			String word = strTok[0];
			int ID = Integer.parseInt(strTok[1]);
			
			idToWord.put(ID, word);
			wordToId.put(word, ID);
		}
		
		rd.close();
		return true;
	}
	
	/***************************/
	//write the dictionary into a word map file, filename
	//Write as word id pairs
	public boolean writeWordFile(String filename) throws IOException{
		//variables
		String word;
		
		BufferedWriter wrt = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
		
		//write the number of words first
		wrt.write(wordToId.size() + "\n");
		
		//write the pairs in the dictionary
		Iterator<String> itr = wordToId.keySet().iterator();
		while(itr.hasNext()){
			word = itr.next();
			wrt.write(word + " " + wordToId.get(word) + "\n");
		}
		
		wrt.close();
		return true;
	}
	
	//------------------------------------------------------
	//Query methods
	//------------------------------------------------------
	/***************************/
	//get the word given the ID
	public String getWord(int ID){
		return idToWord.get(ID);
	}
	
	/***************************/
	//get the ID given the word
	public int getID(String word){
		return wordToId.get(word);
	}
	
	/***************************/
	//check if the hash-map contains the word
	public boolean containsWord(String word){
		return wordToId.containsKey(word);
	}
	
	/***************************/
	//check if the hash-map contains the id
	public boolean containsID(int ID){
		return idToWord.containsKey(ID);
	}
}

/***********************************************************************************************************************************/
//A class that defines a document.

class Document{
	
	//------------------------------------------------------
	//The variables
	//------------------------------------------------------
	public int[] words;		//The ID of each word in the document
	public int length;		//The size of the document

	//------------------------------------------------------
	//The constructor
	//------------------------------------------------------
	/***************************/	
	public Document(int[] words, int length){
		this.length = length;
		
		this.words = new int[length];
		for(int i=0; i<length; i++){
			this.words[i] = words[i];
		}
	}
	
	/***************************/
	public Document(Vector<Integer> doc){
		this.length = doc.size();
		
		this.words = new int[length];
		for(int i=0; i<length ; i++){
			this.words[i] = doc.get(i);
		}
	}
}
