using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Collections.Generic;

public class MovePedestrians : MonoBehaviour {
	public struct TrackInfo
	{
		public int startFrame;
		public int frameCount;
		public List<Vector2> points;
	}


	private List<GameObject> people;
	private List<MovePedestrians.TrackInfo> tracks;
	
	private int currentFrame = 0; //Starting Frame

	void Start() {
		Application.targetFrameRate = 10;

		tracks = new List<MovePedestrians.TrackInfo>();
		people = new List<GameObject>();

		//Set camera position
		Camera mainCam = Camera.main;
		mainCam.transform.position = new Vector3 (0, 100, 0);
		mainCam.transform.localEulerAngles = new Vector3 (90, 0, 0);
		mainCam.transform.localScale = new Vector3 (1, 1, 1);
		mainCam.fieldOfView = 30;
		
		string line;
		char[] delimiterChars = { ' ','\t' };

		//Read file
		using (StreamReader reader = new StreamReader("../Output/HUB3.out"))
		{
			//First section
			do{
				line = reader.ReadLine();
				if (line != ""){
					people.Add(GameObject.CreatePrimitive(PrimitiveType.Cylinder));
					string[] frameInfo = line.Split(delimiterChars);
					int startFrame = Convert.ToInt32(frameInfo[0]);
					int frameLength = Convert.ToInt32(frameInfo[1]);

					TrackInfo track;
					track.startFrame = startFrame;
					track.frameCount = frameLength;
					track.points = new List<Vector2>();
					tracks.Add(track);
				}
			} while (line != "");

			//Second section
			foreach (TrackInfo track in tracks) {
				for (int i = 0; i < track.frameCount; i++) {
					line = reader.ReadLine();
					if (line == null){
						Debug.LogError("File out of bounds.");
					}

					String[] locInfo = line.Split(delimiterChars);

					Vector2 point;
					point.x = float.Parse(locInfo[0]);
					point.y = float.Parse(locInfo[1]);
					track.points.Add(point);
				}
			}
		}
	}	

	//Update each "person" during each frame
	void Update(){
		int personNum = 0;
		foreach (TrackInfo track in tracks) {
			GameObject person = people[personNum] as GameObject;

			if (track.startFrame < currentFrame && currentFrame < (track.startFrame + track.frameCount)) {
				person.SetActive(true);

				int i = currentFrame - track.startFrame;
				Vector2 point = track.points[i];
				float x = point.x/10;
				float y = point.y/10;
				person.transform.position = new Vector3 (x, 0, y);

				Debug.Log("Frame: " + currentFrame + " Person: " + personNum + " X: " + x + " Y: " + y);
			} else {
				person.SetActive(false);
			}

			personNum++;
		}

		currentFrame++;

	}
}


