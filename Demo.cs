using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Collections.Generic;

public class Demo : MonoBehaviour {
	public List<GameObject> people = new List<GameObject>();
	public List<String[]> tracks = new List<String[]> ();
	Camera mainCam;
	int x = 0;
	int z = 0;
	int count = 0;
	string line;
	char[] delimiterChars = { ' ', ',', '.', ':', '\t' };
	void Start(){
		mainCam = Camera.main;
		mainCam.transform.position = new Vector3 (0, 27, 0);
		mainCam.transform.localEulerAngles = new Vector3 (15.5f, 0, 0);
		mainCam.transform.localScale = new Vector3 (1, 1, 1);
		using (StreamReader reader = new StreamReader("Assets/file.txt"))
		{
			do{
				line = reader.ReadLine();
				if (line != null){
					people.Add(GameObject.CreatePrimitive(PrimitiveType.Cylinder));
					string[] words = line.Split(delimiterChars);
					tracks.Add(words);
					count++;
				}
			} while (line != null);
		}
	}	
	void Update(){
		for (int i = 0; i < count; i++) {
			GameObject person = people[i] as GameObject;
			//string[] personTrack = tracks[i] as string[];
			x = Convert.ToInt32 (tracks[i][0]);
			z = Convert.ToInt32 (tracks[i][1]);
			if (person.transform.position.x > x) {
				person.transform.position += new Vector3 (-1, 0, 0);			
			}
			if (person.transform.position.x < x) {
				person.transform.position += new Vector3 (1, 0, 0);			
			}
			if (person.transform.position.z > z) {
				person.transform.position += new Vector3 (0, 0, -1);			
			}
			if (person.transform.position.z < z) {
				person.transform.position += new Vector3 (0, 0, 1);			
			}
		}
	}
}


