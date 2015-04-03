using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Collections.Generic;

public class movePedestrians : MonoBehaviour {
	public List<GameObject> people = new List<GameObject>();
	public List<List<float>> frames = new List<List<float>> ();
	public List<int[]> startEnd = new List<int[]> ();
	public List<float[]> cords = new List<float[]> ();

	Camera mainCam;

	int currentFrame = 1; //Starting Frame
	float x = 0;
	float y = 0;
	int count = 0; // Number of tracks 
	int maxFrames = 10000;

	string line;
	char[] delimiterChars = { ' ','\t' };

	void Start(){
		//Set camera position
		mainCam = Camera.main;
		mainCam.transform.position = new Vector3 (0, 100, 0);
		mainCam.transform.localEulerAngles = new Vector3 (90, 0, 0);
		mainCam.transform.localScale = new Vector3 (1, 1, 1);
		mainCam.fieldOfView = 30;

		//Initialize frames with empty lists
		for (int x = 0; x < maxFrames; x++) {
			frames.Add(new List<float>());
		}

		//Read file
		using (StreamReader reader = new StreamReader("example_file.txt"))
		{
			//First section
			do{
				line = reader.ReadLine();
				if (line != ""){
					people.Add(GameObject.CreatePrimitive(PrimitiveType.Cylinder));
					string[] frameInfo = line.Split(delimiterChars);
					int startFrame = Convert.ToInt32(frameInfo[0]);
					int frameLength = Convert.ToInt32(frameInfo[1]);

					int[] temp = new int[3];
					temp[0] = count;
					temp[1] = startFrame;
					temp[2] = frameLength;
					startEnd.Add(temp);
					count++;
				}
			} while (line != "");

			//Second section
			do{
				line = reader.ReadLine();
				if (line != null){
					String[] locInfo = line.Split(delimiterChars);
					x = float.Parse(locInfo[0]);
					y = float.Parse(locInfo[1]);
					float[] temp = new float[2];
					temp[0] = x;
					temp[1] = y;
					cords.Add(temp);
				}
			} while (line != null);
		}

		//Populate frames list with coordinates
		int totalNumOfFrames = 0;
		for (int i = 0; i < count; i++) { //Loop through each track
			float personNum = (float)startEnd[i][0];
			int start = startEnd[i][1];
			int numOfFrames = startEnd[i][2];

			int tempFrame = start;

			int tempNumOfFrames = totalNumOfFrames;
			totalNumOfFrames += numOfFrames;
			for (int j = tempNumOfFrames; j < totalNumOfFrames; j++){ //Loop through coordinates for current track
				List<float> temp;
				if (frames[tempFrame] == null){
					temp = new List<float>();
				}
				else{
					temp = frames[tempFrame];
				}
				temp.Add(personNum);
				temp.Add(cords[j][0]);
				temp.Add(cords[j][1]);
				frames[tempFrame] = temp;
				tempFrame++;
			}

		}
	}	

	//Update each "person" during each frame
	void Update(){
		var currentTracks = frames[currentFrame];
		for (int i = 0; i < currentTracks.Count; i+=3) {
			int personNum = Convert.ToInt32(currentTracks[i]);
			GameObject person = people[personNum] as GameObject;
			x = currentTracks[i+1];
			y = currentTracks[i+2];
			person.transform.position = new Vector3 (x, 0, y);
			//Debug.Log("Frame: " + currentFrame + " Person: " + personNum + " X: " + x + " Y: " + y);
		}

		currentFrame++;

	}
}


