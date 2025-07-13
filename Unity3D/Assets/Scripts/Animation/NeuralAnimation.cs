using System.Diagnostics;
using UnityEngine;
using DeepLearning;
#if UNITY_EDITOR
using UnityEditor;
#endif

public abstract class NeuralAnimation : MonoBehaviour {

	public NeuralNetwork NeuralNetwork = null;

	// public Actor Actor;
	// public Actor Opponent;
	// public Actor GT;
	// public int countFrame = 0;

	public float inferenceTime {get; private set;}
	public float framerate = 30f;

	protected abstract void Setup();
	protected abstract void Feed();
	protected abstract void Read();
	protected abstract void Feed2();
	protected abstract void Read2();
	protected abstract void OnGUIDerived();
	protected abstract void OnRenderObjectDerived();

	void Reset() {
		NeuralNetwork = GetComponent<NeuralNetwork>();
		// Actor = GetComponent<Actor>();
	}

    void Start() {
		Setup();
    }

	void Update() {
		System.DateTime t = Utility.GetTimestamp();
		Utility.SetFPS(Mathf.RoundToInt(framerate));
		if(NeuralNetwork != null && NeuralNetwork.Setup && !NeuralNetwork.CheckFinishReading()) {
			NeuralNetwork.ResetPivot();
			Feed();

			NeuralNetwork.Predict();

			NeuralNetwork.ResetPivot(); 
			Read();
		}
		NeuralNetwork.countFrame += 1;
		inferenceTime = (float)Utility.GetElapsedTime(t);
	}

    void OnGUI() {
		if(NeuralNetwork != null && NeuralNetwork.Setup) {
			OnGUIDerived();
		}
    }

	void OnRenderObject() {
		if(NeuralNetwork != null && NeuralNetwork.Setup && Application.isPlaying) {
			OnRenderObjectDerived();
		}
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(NeuralAnimation), true)]
	public class NeuralAnimation_Editor : Editor {

		public NeuralAnimation Target;

		void Awake() {
			Target = (NeuralAnimation)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			DrawDefaultInspector();

			EditorGUILayout.HelpBox("Inference Time: " + 1000f*Target.inferenceTime + "ms", MessageType.None);

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}
	#endif

}