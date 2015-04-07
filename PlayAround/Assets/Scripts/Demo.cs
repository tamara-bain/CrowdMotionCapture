using UnityEngine;
using System.Collections;
using System;
using System.IO;

public class Demo : MonoBehaviour {	
	int x = 0;
	int z = 0;
	string line;
	char[] delimiterChars = { ' ', ',', '.', ':', '\t' };
	void Start(){


		using (StreamReader reader = new StreamReader("file.txt"))
		{
			line = reader.ReadLine();
		}
		Console.WriteLine(line);
		string[] words = line.Split(delimiterChars);
		print (words[1]);
		x = Convert.ToInt32 (words[0]);
		z = Convert.ToInt32 (words[1]);

	}	
	void Update(){

		if(transform.position.x>x){
			transform.position+= new Vector3 (-1, 0, 0);			
		}
		if(transform.position.x<x){
			transform.position+= new Vector3 (1, 0, 0);			
		}
		if(transform.position.z>z){
			transform.position+= new Vector3 (0, 0, -1);			
		}
		if(transform.position.z<z){
			transform.position+= new Vector3 (0, 0, 1);			
		}
	}
}


