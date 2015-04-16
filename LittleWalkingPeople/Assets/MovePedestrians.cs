using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;


public class MovePedestrians : MonoBehaviour {
	public float scaling = 0.1f;
	public GameObject prefab;
	
	private struct TrackInfo
	{
		public int startFrame;
		public int frameCount;
		public List<Vector2> points;
	}

	private List<GameObject> people;
	private List<MovePedestrians.TrackInfo> tracks;
	private int currentFrame = 0; //Starting Frame

	private bool loaded;

	//initialize file browser
	private FileBrowser fb = new FileBrowser();
	private string path = "";

	private Camera mainCam;

	void Start() {
		//Application.targetFrameRate = 10;

		tracks = new List<MovePedestrians.TrackInfo>();
		people = new List<GameObject>();

		loaded = false;

		//Set up camera camera position
		mainCam = Camera.main;
		mainCam.transform.position = new Vector3 (30, 20, 0);
		mainCam.transform.localEulerAngles = new Vector3 (40, 0, 0);
		//mainCam.transform.position = new Vector3 (0, 20, 0);
		//mainCam.transform.localEulerAngles = new Vector3 (90, 0, 0);
		mainCam.transform.localScale = new Vector3 (1, 1, 1);
		mainCam.fieldOfView = 30;
	}	

	void OnGUI(){
		if (path == "") {
			GUILayout.BeginHorizontal ();
			GUILayout.Label ("Selected File: " + path);
			GUILayout.EndHorizontal ();
			//draw and display output
			if (fb.draw ()) { //true is returned when a file has been selected
				//the output file is a member if the FileInfo class, if cancel was selected the value is null
				path = (fb.outputFile == null) ? "cancel hit" : fb.outputFile.FullName;
			}
		} else if (!loaded) {
			load();
		}
	}

	void load() {
		string line;
		char[] delimiterChars = { ' ','\t' };

		//Read file
		using (StreamReader reader = new StreamReader(path))
		{
			//First section
			do{
				line = reader.ReadLine();
				if (line != ""){
					//people.Add(GameObject.CreatePrimitive(PrimitiveType.Cylinder));
					people.Add(Instantiate(prefab) as GameObject);
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

		loaded = true;

		// Restart player
		currentFrame = 0;

		// Attach flycam to main camera
		mainCam.gameObject.AddComponent<ExtendedFlycam>();
	}

	//Update each "person" during each frame
	void Update(){
		if (Input.GetKey ("escape")) {
			Application.Quit();
		}

		if (Input.GetKey ("space")) {
			// Restart player
			currentFrame = 0;
		}

		int personNum = 0;
		foreach (TrackInfo track in tracks) {
			GameObject person = people[personNum] as GameObject;

			if (track.startFrame < currentFrame && currentFrame < (track.startFrame + track.frameCount)) {
				person.SetActive(true);

				int i = currentFrame - track.startFrame;
				Vector2 point = track.points[i];
				float x = point.x*scaling;
				float y = point.y*scaling;
				person.transform.position = new Vector3 (x, 0, y);

				//Debug.Log("Frame: " + currentFrame + " Person: " + personNum + " X: " + x + " Y: " + y);
			} else {
				person.SetActive(false);
			}

			personNum++;
		}

		currentFrame++;

	}
}


