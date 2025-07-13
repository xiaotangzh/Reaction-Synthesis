#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class LeftHandModule : Module {

	public enum TOPOLOGY {Biped, Quadruped, Custom};

	public int Root = 0;
	public TOPOLOGY Topology = TOPOLOGY.Biped;
	public int LeftHand;
	public int RightShoulder, LeftShoulder, RightHip, LeftHip, Neck, Hips;
	public LayerMask Ground = 0;
	public Axis ForwardAxis = Axis.ZPositive;
	public bool Smooth = true;
	public enum COLOR {Blue, Red, Orange};
	public COLOR Color = COLOR.Blue;
	public bool DrawDirection = true;
	public enum DRAWHALF {Full, FirstHalf, SecondHalf};
	public DRAWHALF DrawHalf = DRAWHALF.Full;

	//Precomputed
	private float[] SamplingWindow = null;
	private Quaternion[] TmpRotations = null;
	private float[] TmpAngles = null;
	private Precomputable<Matrix4x4>[] PrecomputedRegularTransformations = null;
	private Precomputable<Matrix4x4>[] PrecomputedInverseTransformations = null;
	private Precomputable<Vector3>[] PrecomputedRegularPositions = null;
	private Precomputable<Vector3>[] PrecomputedInversePositions = null;
	private Precomputable<Quaternion>[] PrecomputedRegularRotations = null;
	private Precomputable<Quaternion>[] PrecomputedInverseRotations = null;
	private Precomputable<Vector3>[] PrecomputedRegularVelocities = null;
	private Precomputable<Vector3>[] PrecomputedInverseVelocities = null;

	public override ID GetID() {
		return ID.LeftHand;
	}

	public override void DerivedResetPrecomputation() {
		SamplingWindow = null;
		TmpRotations = null;
		TmpAngles = null;
		PrecomputedRegularTransformations = Data.ResetPrecomputable(PrecomputedRegularTransformations);
		PrecomputedInverseTransformations = Data.ResetPrecomputable(PrecomputedInverseTransformations);
		PrecomputedRegularPositions = Data.ResetPrecomputable(PrecomputedRegularPositions);
		PrecomputedInversePositions = Data.ResetPrecomputable(PrecomputedInversePositions);
		PrecomputedRegularRotations = Data.ResetPrecomputable(PrecomputedRegularRotations);
		PrecomputedInverseRotations = Data.ResetPrecomputable(PrecomputedInverseRotations);
		PrecomputedRegularVelocities = Data.ResetPrecomputable(PrecomputedRegularVelocities);
		PrecomputedInverseVelocities = Data.ResetPrecomputable(PrecomputedInverseVelocities);
	}

	public override ComponentSeries DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored) {
		RootSeries instance = new RootSeries(global);
		for(int i=0; i<instance.Samples.Length; i++) {
			instance.Transformations[i] = GetRootTransformation(timestamp + instance.Samples[i].Timestamp, mirrored);
			instance.Velocities[i] = GetRootVelocity(timestamp + instance.Samples[i].Timestamp, mirrored);
		}
		if(Color == COLOR.Blue) {instance.defaultColor = new Color(0.15f, 0.50f, 0.73f, 1f);}
		else if(Color == COLOR.Red) {instance.defaultColor = new Color(0.74f, 0.24f, 0.33f, 1f);}
		else {instance.defaultColor = UltiDraw.Orange.Opacity(0.75f);}

		if(DrawHalf == DRAWHALF.FirstHalf) {instance.drawHalf=1;}
		else if(DrawHalf == DRAWHALF.SecondHalf) {instance.drawHalf=2;}

		instance.DrawDirection = DrawDirection;
		return instance;
	}

	protected override void DerivedInitialize() {
		MotionData.Hierarchy.Bone x = Data.Source.FindBoneContains("LeftHand");
		Root = x == null ? 0 : x.Index;

		MotionData.Hierarchy.Bone rs = Data.Source.FindBoneContains("RightShoulder");
		RightShoulder = rs == null ? 0 : rs.Index;
		MotionData.Hierarchy.Bone ls = Data.Source.FindBoneContains("LeftShoulder");
		LeftShoulder = ls == null ? 0 : ls.Index;
		MotionData.Hierarchy.Bone rh = Data.Source.FindBoneContains("RightHip", "RightUpLeg");
		RightHip = rh == null ? 0 : rh.Index;
		MotionData.Hierarchy.Bone lh = Data.Source.FindBoneContains("LeftHip", "LeftUpLeg");
		LeftHip = lh == null ? 0 : lh.Index;
		MotionData.Hierarchy.Bone n = Data.Source.FindBoneContains("Neck");
		Neck = n == null ? 0 : n.Index;
		MotionData.Hierarchy.Bone h = Data.Source.FindBoneContains("Hips");
		Hips = h == null ? 0 : h.Index;
		Ground = LayerMask.GetMask("Ground");
	}

	protected override void DerivedLoad(MotionEditor editor) {
		
	}

	protected override void DerivedCallback(MotionEditor editor) {
		Frame frame = editor.GetCurrentFrame();
		editor.GetActor().transform.OverridePositionAndRotation(GetRootPosition(frame.Timestamp, editor.Mirror), GetRootRotation(frame.Timestamp, editor.Mirror));
	}

	protected override void DerivedGUI(MotionEditor editor) {
	
	}

	protected override void DerivedDraw(MotionEditor editor) {

	}

	protected override void DerivedInspector(MotionEditor editor) {
		Root = EditorGUILayout.Popup("Root", Root, Data.Source.GetBoneNames());
		Topology = (TOPOLOGY)EditorGUILayout.EnumPopup("Topology", Topology);
		RightShoulder = EditorGUILayout.Popup("Right Shoulder", RightShoulder, Data.Source.GetBoneNames());
		LeftShoulder = EditorGUILayout.Popup("Left Shoulder", LeftShoulder, Data.Source.GetBoneNames());
		RightHip = EditorGUILayout.Popup("Right Hip", RightHip, Data.Source.GetBoneNames());
		LeftHip = EditorGUILayout.Popup("Left Hip", LeftHip, Data.Source.GetBoneNames());
		Neck = EditorGUILayout.Popup("Neck", Neck, Data.Source.GetBoneNames());
		Hips = EditorGUILayout.Popup("Hips", Hips, Data.Source.GetBoneNames());
		ForwardAxis = (Axis)EditorGUILayout.EnumPopup("Forward Axis", ForwardAxis);
		Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Ground), InternalEditorUtility.layers));
		Smooth = EditorGUILayout.Toggle("Smooth", Smooth);
		Color = (COLOR)EditorGUILayout.EnumPopup("Color", Color);
		DrawDirection = EditorGUILayout.Toggle("DrawDirection", DrawDirection);
		DrawHalf = (DRAWHALF)EditorGUILayout.EnumPopup("DrawHalf", DrawHalf);
	}

	public Matrix4x4 GetRootTransformation(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseTransformations[index] == null) {
				PrecomputedInverseTransformations[index] = new Precomputable<Matrix4x4>(Compute());
			}
			if(!mirrored && PrecomputedRegularTransformations[index] == null) {
				PrecomputedRegularTransformations[index] = new Precomputable<Matrix4x4>(Compute());
			}
			return mirrored ? PrecomputedInverseTransformations[index].Value : PrecomputedRegularTransformations[index].Value;
		}

		return Compute();
		Matrix4x4 Compute() {
			return Matrix4x4.TRS(GetRootPosition(timestamp, mirrored), GetRootRotation(timestamp, mirrored), Vector3.one);
		}
	}

	public Vector3 GetRootPosition(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInversePositions[index] == null) {
				PrecomputedInversePositions[index] = new Precomputable<Vector3>(Compute());
			}
			if(!mirrored && PrecomputedRegularPositions[index] == null) {
				PrecomputedRegularPositions[index] = new Precomputable<Vector3>(Compute());
			}
			return mirrored ? PrecomputedInversePositions[index].Value : PrecomputedRegularPositions[index].Value;
		}

		return Compute();
		Vector3 Compute() {
			return RootPosition(timestamp, mirrored);
		}
	}

	public Quaternion GetRootRotation(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseRotations[index] == null) {
				PrecomputedInverseRotations[index] = new Precomputable<Quaternion>(Compute());
			}
			if(!mirrored && PrecomputedRegularRotations[index] == null) {
				PrecomputedRegularRotations[index] = new Precomputable<Quaternion>(Compute());
			}
			return mirrored ? PrecomputedInverseRotations[index].Value : PrecomputedRegularRotations[index].Value;
		}

		return Compute();
		Quaternion Compute() {
			if(!Smooth)  {
				return RootRotation(timestamp, mirrored);
			}
			
			SamplingWindow = SamplingWindow == null ? Data.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f) : SamplingWindow;
			TmpRotations = TmpRotations.Validate(SamplingWindow.Length);
			TmpAngles = TmpAngles.Validate(TmpRotations.Length-1);
			for(int i=0; i<SamplingWindow.Length; i++) {
				TmpRotations[i] = RootRotation(timestamp + SamplingWindow[i], mirrored);
			}
			for(int i=0; i<TmpAngles.Length; i++) {
				TmpAngles[i] = Vector3.SignedAngle(TmpRotations[i].GetForward(), TmpRotations[i+1].GetForward(), Vector3.up) / (SamplingWindow[i+1] - SamplingWindow[i]);
			}
			float power = Mathf.Deg2Rad*Mathf.Abs(TmpAngles.Gaussian());

			return TmpRotations.Gaussian(power);
		}
	}

	public Vector3 GetRootVelocity(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseVelocities[index] == null) {
				PrecomputedInverseVelocities[index] = new Precomputable<Vector3>(Compute());
			}
			if(!mirrored && PrecomputedRegularVelocities[index] == null) {
				PrecomputedRegularVelocities[index] = new Precomputable<Vector3>(Compute());
			}
			return mirrored ? PrecomputedInverseVelocities[index].Value : PrecomputedRegularVelocities[index].Value;
		}
		
		return Compute();
		Vector3 Compute() {
			return (GetRootPosition(timestamp, mirrored) - GetRootPosition(timestamp - Data.GetDeltaTime(), mirrored)) / Data.GetDeltaTime();
		}
	}

	private Vector3 RootPosition(float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
			float boundary = Mathf.Clamp(timestamp, start, end);
			float pivot = 2f*boundary - timestamp;
			float clamped = Mathf.Clamp(pivot, start, end);
			return 2f*RootPosition(Data.GetFrame(boundary)) - RootPosition(Data.GetFrame(clamped));
		} else {
			return RootPosition(Data.GetFrame(timestamp));
		}

		Vector3 RootPosition(Frame frame) {
			// Vector3 position = frame.GetBoneTransformation(Root, mirrored).GetPosition();
			Vector3 position = frame.GetBoneTransformation(Root, mirrored).GetPosition();
			return position;
		}
	}

	private Quaternion RootRotation(float timestamp, bool mirrored) {
		float start = Data.GetFirstValidFrame().Timestamp;
		float end = Data.GetLastValidFrame().Timestamp;
		if(timestamp < start || timestamp > end) {
			float boundary = Mathf.Clamp(timestamp, start, end);
			float pivot = 2f*boundary - timestamp;
			float clamped = Mathf.Clamp(pivot, start, end);
			return RootRotation(Data.GetFrame(clamped));
		} else {
			return RootRotation(Data.GetFrame(timestamp));
		}

		Quaternion RootRotation(Frame frame) {
			if(Topology == TOPOLOGY.Biped) {
				Quaternion rotation = frame.GetBoneTransformation(Root, mirrored).GetRotation();
				return rotation;
			}
			if(Topology == TOPOLOGY.Quadruped) {
				Quaternion rotation = frame.GetBoneTransformation(Root, mirrored).GetRotation();
				return rotation;
			}
			if(Topology == TOPOLOGY.Custom) {
				Quaternion rotation = frame.GetBoneTransformation(Root, mirrored).GetRotation();
				return rotation;
			}
			return Quaternion.identity;
		}
	}

}
#endif
