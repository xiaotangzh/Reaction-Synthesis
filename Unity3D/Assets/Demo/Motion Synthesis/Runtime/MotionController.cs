using UnityEngine;
using DeepLearning;
using System;
using System.IO;
using UnityEditor;
using UnityEngine.XR;
using System.Collections.Generic;
using System.Linq; // List<> OrderBy();

public class MotionController : NeuralAnimation {
	// player2
    public Actor player1;
    public Actor player2;
    public Actor gtPlayer1;
    public Actor gtPlayer2;
	private Actor controlledPlayer;
	public Actor testPlayer;

	// control
	public GameObject arrow;
	public bool enableControl = false;
	public bool drawHistories = false;
	public enum controlPlayers {player1, player2};
	public controlPlayers controllPlayer;

	public enum controlFeatures {distance, angle};
	public controlFeatures controlFeature;

	// sampling
	public bool drawSamples = false;
	public int player2Samples = 1;
	public GameObject samplesObject;
	
	// switches
	[System.Flags]
	public enum DrawRootSeries
	{
		None = 0,
		player1 = 1 << 0, // 1
		player2 = 1 << 1, // 2
	}
	public DrawRootSeries drawRootSeries;
	[System.Flags]
	public enum DrawJointSeries
	{
		None = 0,
		player1 = 1 << 0, // 1
		player2 = 1 << 1, // 2
	}
	public DrawJointSeries drawJointSeries;
	public bool enableGroundTruthPlayer = true;
	public bool enableTestPlayer = true;
	public bool disableRenderer = false;
	public bool enableGtIntention = false;
	public bool drawContact = false;
	// public bool player1Matching = false;

	// trajectories
	public TimeSeries timeSeries;
	public RootSeries rootSeries;
	public RootSeries headSeries;
	public RootSeries leftHandSeries;
	public RootSeries rightHandSeries;
	public RootSeries leftLegSeries;
	public RootSeries rightLegSeries;
	public RootSeries controlSeries;

    public RootSeries rootSeries2;
	public RootSeries headSeries2;
	public RootSeries leftHandSeries2;
	public RootSeries rightHandSeries2;
	public RootSeries leftLegSeries2;
	public RootSeries rightLegSeries2;
	public ContactSeries contactSeries;
	public ContactSeries contactSeries2;
	public RootSeries gtRootSeries;
	public RootSeries gtRootSeries2;


	public RootSeries testSeries;
	public List<List<Vector3>> historySeries = new List<List<Vector3>>();
	
	// others
	public Controller.TYPE ControlType = Controller.TYPE.Keyboard;
	private Controller controller;
	public Camera camera = null;
	private List<State> states = new List<State>();
	private State newState;

	// ik
	public bool EnableHandIK = false;
	public float HandIKMin = 0f;
	public float HandIKMax = 0f;
	public bool EnableFootIK = false;
	public float FootIKMax = 0f;
	private UltimateIK.Model LeftFootIK;
	private UltimateIK.Model RightFootIK;
	private UltimateIK.Model LeftHandIK;
	private UltimateIK.Model RightHandIK;

	private Camera GetCamera() {
		return camera == null ? Camera.main : camera;
	}

	protected override void Setup() {
		if(disableRenderer==true) {
			MeshRenderer[] renderers = GetComponentsInChildren<MeshRenderer>();
			foreach (MeshRenderer renderer in renderers) {
				renderer.enabled = false;
			}
		}

		if(samplesObject==null) {samplesObject = GameObject.Find("Samples");}

		if(player2Samples<=1 || NeuralNetwork.enableSocket==false) {
			samplesObject.SetActive(false);
		}

		if(NeuralNetwork.enableSocket==false || enableGroundTruthPlayer==false) {
			gtPlayer1.gameObject.SetActive(false);
			gtPlayer2.gameObject.SetActive(false);
		}
		else if(NeuralNetwork.enableSocket==true && enableGroundTruthPlayer==true) {
			gtPlayer1.gameObject.SetActive(true);
			gtPlayer2.gameObject.SetActive(true);
		}

		controller = new Controller();
		controller.ControlType = ControlType;

		InitializeSeries();

		// inverse kinematic
		LeftFootIK = UltimateIK.BuildModel(player2.FindTransform("LeftUpLeg"), player2.GetBoneTransforms("LeftFoot"));
		RightFootIK = UltimateIK.BuildModel(player2.FindTransform("RightUpLeg"), player2.GetBoneTransforms("RightFoot"));
		LeftHandIK = UltimateIK.BuildModel(player2.FindTransform("LeftShoulder"), player2.GetBoneTransforms("LeftHand"));
		RightHandIK = UltimateIK.BuildModel(player2.FindTransform("RightShoulder"), player2.GetBoneTransforms("RightHand"));
	}

	protected override void Feed() {
		ControlArrow();
		Control();

		// FeedChar1();
		// FeedChar2();

		Matrix4x4 Root = player1.GetRoot().GetWorldMatrix(true);
		Matrix4x4 Root2 = player2.GetRoot().GetWorldMatrix(true);

		// char 1 traj
		FeedRootTrajectory(rootSeries, Root2, 0, timeSeries.KeyCount);
		FeedJointTrajectory(headSeries, Root2, 0, timeSeries.KeyCount);
		FeedJointTrajectory(leftHandSeries, Root2, 0, timeSeries.KeyCount);
		FeedJointTrajectory(rightHandSeries, Root2, 0, timeSeries.KeyCount);
		FeedJointTrajectory(leftLegSeries, Root2, 0, timeSeries.KeyCount);
		FeedJointTrajectory(rightLegSeries, Root2, 0, timeSeries.KeyCount);
		// char 2 traj
		FeedRootTrajectory(rootSeries2, Root2, 0, timeSeries.PivotKey+1);
		FeedJointTrajectory(headSeries2, Root2, 0, timeSeries.PivotKey+1);
		FeedJointTrajectory(leftHandSeries2, Root2, 0, timeSeries.PivotKey+1);
		FeedJointTrajectory(rightHandSeries2, Root2, 0, timeSeries.PivotKey+1);
		FeedJointTrajectory(leftLegSeries2, Root2, 0, timeSeries.PivotKey+1);
		FeedJointTrajectory(rightLegSeries2, Root2, 0, timeSeries.PivotKey+1);
		// char2 pose
		FeedPose(player2, Root2);

		// motion matching for both
		FeedRootTrajectory(controlSeries, gtPlayer1.GetRoot().GetWorldMatrix(true), timeSeries.PivotKey+1, timeSeries.KeyCount);
		FeedPose(gtPlayer1, gtPlayer1.GetRoot().GetWorldMatrix(true));
		FeedRootTrajectory(gtRootSeries2, gtPlayer2.GetRoot().GetWorldMatrix(true), timeSeries.PivotKey+1, timeSeries.KeyCount);
		FeedPose(gtPlayer2, gtPlayer2.GetRoot().GetWorldMatrix(true));

		// char1 intention
		FeedJointTrajectory(headSeries, Root, 0, timeSeries.KeyCount);
		FeedJointTrajectory(leftHandSeries, Root, 0, timeSeries.KeyCount);
		FeedJointTrajectory(rightHandSeries, Root, 0, timeSeries.KeyCount);
		FeedJointTrajectory(leftLegSeries, Root, 0, timeSeries.KeyCount);
		FeedJointTrajectory(rightLegSeries, Root, 0, timeSeries.KeyCount);
	}

	protected override void Read() {
		// if(NeuralNetwork.countFrame==0) {CreateHistoryTrajectory();}
		// if(enableTestPlayer) {ReadTestChar(player1, testPlayer);}
		// if(NeuralNetwork.enableSocket && enableGroundTruthPlayer) {
		// 	ReadGT(gtPlayer1, rootSeries, headSeries, leftHandSeries, rightHandSeries, leftLegSeries, rightLegSeries, contactSeries); 
		// 	ReadGT(gtPlayer2, rootSeries2, headSeries2, leftHandSeries2, rightHandSeries2, leftLegSeries2, rightLegSeries2, contactSeries2); 
		// }
		// else if(NeuralNetwork.enableSocket && enableGroundTruthPlayer==false) {
		// 	ReadGT(null); 
		// 	ReadGT(null);
		// }
        // ReadChar1();
		// ReadChar2();

		if(NeuralNetwork.enableSocket==false || NeuralNetwork.countFrame<NeuralNetwork.initHistory) {
			ReadGT(player1, rootSeries, headSeries, leftHandSeries, rightHandSeries, leftLegSeries, rightLegSeries, contactSeries); 
			ReadGT(player2, rootSeries2, headSeries2, leftHandSeries2, rightHandSeries2, leftLegSeries2, rightLegSeries2, contactSeries2); 
		} else {
			ReadGT(player1, rootSeries, headSeries, leftHandSeries, rightHandSeries, leftLegSeries, rightLegSeries, contactSeries); 
			ReadPrediction(player2, rootSeries2, headSeries2, leftHandSeries2, rightHandSeries2, leftLegSeries2, rightLegSeries2, contactSeries2);
			if(drawSamples) ReadPoseSamples(player2);
		}
		if(NeuralNetwork.enableSocket==true && enableGroundTruthPlayer) {
			ReadGT(gtPlayer1, gtRootSeries, testSeries, testSeries, testSeries, testSeries, testSeries, null); 
			ReadGT(gtPlayer2, gtRootSeries2, testSeries, testSeries, testSeries, testSeries, testSeries, null);
		} 

		if(EnableHandIK) {
			ProcessHandContact(LeftHandIK, player1.FindBone("RightHand").Transform, DetectHandContact(LeftHandIK, player1.FindBone("RightHand").Transform.position, HandIKMin, HandIKMax)); 
			ProcessHandContact(RightHandIK, player1.FindBone("LeftHand").Transform, DetectHandContact(RightHandIK, player1.FindBone("LeftHand").Transform.position, HandIKMin, HandIKMax)); 
		}
		if(EnableFootIK) {
			ProcessFootContact(LeftFootIK, DetectFootContact(LeftFootIK, 0f, FootIKMax));
			ProcessFootContact(RightFootIK, DetectFootContact(RightFootIK, 0f, FootIKMax));
		}
		CorrectTwist(player2);
	}

	private void FeedRootTrajectory(RootSeries rootSeries, Matrix4x4 To, int startKey, int endKey) {
		for(int i=startKey; i<endKey; i++) {
			int index = timeSeries.GetKey(i).Index;
			NeuralNetwork.FeedXZ(rootSeries.GetPosition(index).GetRelativePositionTo(To));
			NeuralNetwork.FeedXZ(rootSeries.GetDirection(index).normalized.GetRelativeDirectionTo(To));
			NeuralNetwork.FeedXZ(rootSeries.GetVelocity(index).GetRelativeDirectionTo(To));
		}
	}
	private void FeedJointTrajectory(RootSeries series, Matrix4x4 To, int startKey, int endKey, bool withDirectionVelocity=true) {
		for(int i=startKey; i<endKey; i++) {
			int index = timeSeries.GetKey(i).Index;
			NeuralNetwork.Feed(series.GetPosition(index).GetRelativePositionTo(To));
			if(withDirectionVelocity) {
				NeuralNetwork.Feed(series.GetDirection(index).normalized.GetRelativeDirectionTo(To));
				NeuralNetwork.Feed(series.GetVelocity(index).GetRelativeDirectionTo(To));
			}
		}
	}
	private void FeedPose(Actor character, Matrix4x4 To) {
		for(int j=0; j<character.Bones.Length; j++) {
			NeuralNetwork.Feed(character.Bones[j].Transform.position.GetRelativePositionTo(To));
			NeuralNetwork.Feed(character.Bones[j].Transform.forward.normalized.GetRelativeDirectionTo(To));
			NeuralNetwork.Feed(character.Bones[j].Transform.up.normalized.GetRelativeDirectionTo(To));
			NeuralNetwork.Feed(character.Bones[j].Velocity.GetRelativeDirectionTo(To));
		}
	}
	private void FeedContact(ContactSeries contact, int startKey, int endKey) {
		for(int i=startKey; i<endKey; i++) {
			int index = timeSeries.GetKey(i).Index;
			NeuralNetwork.Feed(contact.Values[index]);
		}
	}

	private void ReadGT(Actor character, RootSeries roots, RootSeries head, RootSeries leftHand, RootSeries rightHand, RootSeries leftLeg, RootSeries rightLeg, ContactSeries contact) {
		Vector3[] positions = new Vector3[character.Bones.Length];
		Vector3[] forwards = new Vector3[character.Bones.Length];
		Vector3[] upwards = new Vector3[character.Bones.Length];
		Vector3[] velocities = new Vector3[character.Bones.Length];
		Matrix4x4 root;

		Vector3 offset =  NeuralNetwork.ReadVector3();
		Vector3 root_pos = NeuralNetwork.ReadVector3();
		Vector3 root_dir = NeuralNetwork.ReadVector3();
		if(NeuralNetwork.countFrame<NeuralNetwork.initHistory) root = Matrix4x4.TRS(root_pos, Quaternion.LookRotation(root_dir), Vector3.one); // global root
		else root = character.GetRoot().GetWorldMatrix(true) * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one); // update root

		// root series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			roots.Transformations[index] = m;
			roots.Velocities[index] = NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root);
		}

		// head series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			head.Transformations[index] = m;
			head.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
		}

		// left hand series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			leftHand.Transformations[index] = m;
			leftHand.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
		}

		// right hand series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			rightHand.Transformations[index] = m;
			rightHand.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
		}

		// left leg series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			leftLeg.Transformations[index] = m;
			leftLeg.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
		}

		// right leg series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			rightLeg.Transformations[index] = m;
			rightLeg.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);
		}

		// contact
		// float[] contacts = NeuralNetwork.Read(contact.Bones.Length, 0f, 1f);
		// for(int j=0; j<contact.Bones.Length; j++) {
		// 	contact.Values[timeSeries.GetPivot().Index][j] = contacts[j]; //.SmoothStep(ContactPower, BoneContactThreshold);
		// }

		// pose
		for (int i = 0; i < character.Bones.Length; i++) {
			Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

			positions[i] = position;
			forwards[i] = forward;
			upwards[i] = upward;
			velocities[i] = velocity;
		}

		// assign pose
		if(character!=null) {
			character.transform.position = root.GetPosition();
			character.transform.rotation = root.GetRotation();
			for(int i=0; i<character.Bones.Length; i++) {
				character.Bones[i].Transform.position = positions[i];
				character.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
				character.Bones[i].Velocity = velocities[i];
			}
		}

		// assign pivot key of series
		roots.Transformations[timeSeries.Pivot] = character.transform.GetWorldMatrix(true);
		head.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("Head").Index].Transform.GetWorldMatrix(true);
		leftHand.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("LeftHand").Index].Transform.GetWorldMatrix(true);
		rightHand.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("RightHand").Index].Transform.GetWorldMatrix(true);
		leftLeg.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("LeftLeg").Index].Transform.GetWorldMatrix(true);
		rightLeg.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("RightLeg").Index].Transform.GetWorldMatrix(true);
	}

	private void ReadPrediction(Actor character, RootSeries roots, RootSeries head, RootSeries leftHand, RootSeries rightHand, RootSeries leftLeg, RootSeries rightLeg, ContactSeries contact) {
		Vector3[] positions = new Vector3[character.Bones.Length];
		Vector3[] forwards = new Vector3[character.Bones.Length];
		Vector3[] upwards = new Vector3[character.Bones.Length];
		Vector3[] velocities = new Vector3[character.Bones.Length];
		Matrix4x4 root = character.transform.GetWorldMatrix(true);

		// root series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			roots.Transformations[index] = m;
			roots.Velocities[index] = NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root);
		}

		// head series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			// Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			// head.Transformations[index] = m;
			// head.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

			// only position
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.identity, Vector3.one);
			head.Transformations[index] = m;
		}

		// left hand series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			// Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			// leftHand.Transformations[index] = m;
			// leftHand.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

			// only position
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.identity, Vector3.one);
			leftHand.Transformations[index] = m;
		}

		// right hand series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			// Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			// rightHand.Transformations[index] = m;
			// rightHand.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

			// only position
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.identity, Vector3.one);
			rightHand.Transformations[index] = m;
		}

		// left leg series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			// Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			// leftLeg.Transformations[index] = m;
			// leftLeg.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

			// only position
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.identity, Vector3.one);
			leftLeg.Transformations[index] = m;
		}

		// right leg series
		for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			// Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			// rightLeg.Transformations[index] = m;
			// rightLeg.Velocities[index] = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

			// only position
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.identity, Vector3.one);
			rightLeg.Transformations[index] = m;
		}

		// update root
		Vector3 offset =  NeuralNetwork.ReadVector3();
		root = character.GetRoot().GetWorldMatrix(true) * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);

		// contact
		// float[] contacts = NeuralNetwork.Read(contact.Bones.Length, 0f, 1f);
		// for(int j=0; j<contact.Bones.Length; j++) {
		// 	contact.Values[timeSeries.GetPivot().Index][j] = contacts[j]; //.SmoothStep(ContactPower, BoneContactThreshold);
		// }

		// pose
		for (int i = 0; i < character.Bones.Length; i++) {
			Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

			// positions[i] = position;
			positions[i] = Vector3.Lerp(character.Bones[i].Transform.position + velocity / 30, position, 0.5f);
			// positions[i] = position;
			forwards[i] = forward;
			upwards[i] = upward;
			velocities[i] = velocity;
		}

		// assign pose
		if(character!=null) {
			character.transform.position = root.GetPosition();
			character.transform.rotation = root.GetRotation();
			for(int i=0; i<character.Bones.Length; i++) {
				character.Bones[i].Transform.position = positions[i];
				character.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
				character.Bones[i].Velocity = velocities[i];
			}
		}

		// assign pivot key of series
		roots.Transformations[timeSeries.Pivot] = character.transform.GetWorldMatrix(true);
		head.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("Head").Index].Transform.GetWorldMatrix(true);
		leftHand.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("LeftHand").Index].Transform.GetWorldMatrix(true);
		rightHand.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("RightHand").Index].Transform.GetWorldMatrix(true);
		leftLeg.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("LeftLeg").Index].Transform.GetWorldMatrix(true);
		rightLeg.Transformations[timeSeries.Pivot] = character.Bones[character.FindBone("RightLeg").Index].Transform.GetWorldMatrix(true);
	}

	private void ReadPoseSamples(Actor character) {
		Vector3[] positions = new Vector3[character.Bones.Length];
		Vector3[] forwards = new Vector3[character.Bones.Length];
		Vector3[] upwards = new Vector3[character.Bones.Length];
		Vector3[] velocities = new Vector3[character.Bones.Length];
		Matrix4x4 root = character.GetRoot().GetWorldMatrix(true);

		// samples
		for(int s=1; s<=(player2Samples-1); s++) {
			for (int i = 0; i < character.Bones.Length; i++) {
				Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
				Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
				Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
				Vector3 velocity = NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root);

				// positions[i] = position;
				// positions[i] = Vector3.Lerp(samplePlayer.Bones[i].Transform.position + velocity / 30, position, 0.5f);
				positions[i] = position;
				forwards[i] = forward;
				upwards[i] = upward;
				velocities[i] = velocity;
			}

			if(drawSamples) {
				Actor samplePlayer = GameObject.Find("Samples").transform.Find(s.ToString()).GetComponent<Actor>();

				// assign pose
				// samplePlayer.transform.position = root.GetPosition();
				// samplePlayer.transform.rotation = root.GetRotation();
				for(int i=0; i<samplePlayer.Bones.Length; i++) {
					samplePlayer.Bones[i].Transform.position = positions[i];
					samplePlayer.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
					samplePlayer.Bones[i].Velocity = velocities[i];
				}
			}
		}
	}

	private void ReadTestChar(Actor InputChar, Actor character) {
		Vector3[] positions = new Vector3[character.Bones.Length];
		Vector3[] forwards = new Vector3[character.Bones.Length];
		Vector3[] upwards = new Vector3[character.Bones.Length];
		Matrix4x4 root;
		root = InputChar.transform.GetWorldMatrix(true);

		for(int i=0; i<timeSeries.KeyCount; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
			testSeries.Transformations[index] = m;
		}
		
		// joint
		for(int i=0; i<timeSeries.PivotKey*5; i++) {
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
		}

		for(int i=0; i<character.Bones.Length; i++) {
			Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);

			positions[i] = position;
			forwards[i] = forward;
			upwards[i] = upward;
		}

		for(int i=0; i<character.Bones.Length; i++) {
			character.Bones[i].Transform.position = positions[i];
			character.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
		}
	}

    private void ReadChar2() {
        //Read Posture
		Vector3[] positions = new Vector3[player2.Bones.Length];
		Vector3[] forwards = new Vector3[player2.Bones.Length];
		Vector3[] upwards = new Vector3[player2.Bones.Length];
		Vector3[] velocities = new Vector3[player2.Bones.Length];

		Matrix4x4 root;

		if(NeuralNetwork.requireInitPose==true && NeuralNetwork.countFrame==0) {
			Vector3 offset =  NeuralNetwork.ReadVector3();
			Vector3 root_pos = NeuralNetwork.ReadVector3();
			Vector3 root_dir = NeuralNetwork.ReadVector3();
			root = Matrix4x4.TRS(root_pos, Quaternion.LookRotation(root_dir), Vector3.one);
			rootSeries2.Transformations[timeSeries.Pivot] = root;

			// trajectory
			for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
				int index = timeSeries.GetKey(i).Index;
				Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
				rootSeries2.Transformations[index] = m;
			}

			// pose in ego space
			root = rootSeries2.Transformations[timeSeries.Pivot];
			for (int i = 0; i < player2.Bones.Length; i++) {
				Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
				Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
				Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
				positions[i] = position;
				forwards[i] = forward;
				upwards[i] = upward;
			}
		}
		else if (NeuralNetwork.countFrame<NeuralNetwork.initHistory || NeuralNetwork.enableSocket==false){
			Vector3 offset =  NeuralNetwork.ReadVector3();
			Vector3 root_pos = NeuralNetwork.ReadVector3();
			Vector3 root_dir = NeuralNetwork.ReadVector3();
			root = player2.GetRoot().GetWorldMatrix(true) * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
			rootSeries2.Transformations[timeSeries.Pivot] = root;

			// trajectory
			for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
				int index = timeSeries.GetKey(i).Index;
				Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().GetRelativePositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root), Vector3.up), Vector3.one);
				rootSeries2.Transformations[index] = m;
				// RootSeries.Velocities[index] = NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(root);
			}

			// pose in ego space
			root = rootSeries2.Transformations[timeSeries.Pivot];
			for (int i = 0; i < player2.Bones.Length; i++) {
				Vector3 position = NeuralNetwork.ReadVector3().GetRelativePositionFrom(root);
				Vector3 forward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
				Vector3 upward = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
				positions[i] = position;
				forwards[i] = forward;
				upwards[i] = upward;
			}
		}
		else {
			states.Clear();
			for(int k=0; k<player2Samples; k++) {
				// char1 joint trajectory
				Matrix4x4 to = enableGtIntention ? player1.transform.GetWorldMatrix(true) : player2.transform.GetWorldMatrix(true);
				headSeries = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, to, headSeries);
				leftHandSeries = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, to, leftHandSeries);
				rightHandSeries = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, to, rightHandSeries);
				leftLegSeries = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, to, leftLegSeries);
				rightLegSeries = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, to, rightLegSeries);

				// root update
				Vector3 offset =  NeuralNetwork.ReadVector3();
				Matrix4x4 UpdatedRoot = player2.GetRoot().GetWorldMatrix(true) * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
				rootSeries2.Transformations[timeSeries.Pivot] = UpdatedRoot;
				// root trajectory
				for(int i=timeSeries.PivotKey+1; i<timeSeries.KeyCount; i++) {
					int index = timeSeries.GetKey(i).Index;
					Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().GetRelativePositionFrom(UpdatedRoot), Quaternion.LookRotation(NeuralNetwork.ReadXZ().GetRelativeDirectionFrom(UpdatedRoot), Vector3.up), Vector3.one);
					rootSeries2.Transformations[index] = m;
				}

				// joint trajectory
				if(NeuralNetwork.countFrame>=NeuralNetwork.initHistory && NeuralNetwork.enableSocket==true){
					headSeries2 = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, UpdatedRoot, headSeries2);
					leftHandSeries2 = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, UpdatedRoot, leftHandSeries2);
					rightHandSeries2 = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, UpdatedRoot, rightHandSeries2);
					leftLegSeries2 = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, UpdatedRoot, leftLegSeries2);
					rightLegSeries2 = ReadJointTrajectory(timeSeries.PivotKey+1, timeSeries.KeyCount, UpdatedRoot, rightLegSeries2);
				}

				//pose
				for(int b=0; b<player2.Bones.Length; b++) {
					positions[b] = NeuralNetwork.ReadVector3().GetRelativePositionFrom(UpdatedRoot);
					forwards[b] = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(UpdatedRoot);
					upwards[b] = NeuralNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(UpdatedRoot);
				}

				State candidate = new State(player2, UpdatedRoot, positions, forwards, upwards, rootSeries2, arrow, (int)controlFeature, headSeries, leftHandSeries, rightHandSeries, leftLegSeries, rightLegSeries, headSeries2, leftHandSeries2, rightHandSeries2, leftLegSeries2, rightLegSeries2);

				states.Add(candidate);
			}
			float originalDistance = states[0].Distance;
			states = states.OrderBy(s => s.Distance).ToList();
			float newDistance = states[0].Distance;
			// Debug.Log(newDistance<originalDistance ? "yes" : "no");
			newState = states[0];
			root = newState.UpdatedRoot;
			rootSeries2 = newState.RootSeries;
			positions = newState.Positions;
			forwards = newState.Forwards;
			upwards = newState.Upwards;

			headSeries = newState.HeadSeries;
			leftHandSeries = newState.LeftHandSeries;
			rightHandSeries = newState.RightHandSeries;
			leftLegSeries = newState.LeftLegSeries;
			rightLegSeries = newState.RightLegSeries;

			headSeries2 = newState.HeadSeries2;
			leftHandSeries2 = newState.LeftHandSeries2;
			rightHandSeries2 = newState.RightHandSeries2;
			leftLegSeries2 = newState.LeftLegSeries2;
			rightLegSeries2 = newState.RightLegSeries2;
		}

		//Assign Posture
		player2.transform.position = rootSeries2.GetPosition(timeSeries.Pivot);
		player2.transform.rotation = rootSeries2.GetRotation(timeSeries.Pivot);
		for(int i=0; i<player2.Bones.Length; i++) {
			player2.Bones[i].Transform.position = positions[i];
			player2.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
		}

		//joint history
		headSeries2.Transformations[timeSeries.Pivot] = player2.Bones[player2.FindBone("Head").Index].Transform.GetWorldMatrix(true);
		leftHandSeries2.Transformations[timeSeries.Pivot] = player2.Bones[player2.FindBone("LeftHand").Index].Transform.GetWorldMatrix(true);
		rightHandSeries2.Transformations[timeSeries.Pivot] = player2.Bones[player2.FindBone("RightHand").Index].Transform.GetWorldMatrix(true);
		leftLegSeries2.Transformations[timeSeries.Pivot] = player2.Bones[player2.FindBone("LeftLeg").Index].Transform.GetWorldMatrix(true);
		rightLegSeries2.Transformations[timeSeries.Pivot] = player2.Bones[player2.FindBone("RightLeg").Index].Transform.GetWorldMatrix(true);
    }

	private RootSeries ReadJointTrajectory(int start, int end, Matrix4x4 from, RootSeries series, bool assign=true) {
		for(int i=start; i<end; i++) {
			int index = timeSeries.GetKey(i).Index;
			Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadVector3().GetRelativePositionFrom(from), Quaternion.LookRotation(NeuralNetwork.ReadVector3().GetRelativeDirectionFrom(from), Vector3.up), Vector3.one);
			if(assign==true) {series.Transformations[index] = m;}
		}
		return series;
	}
	
	private void ControlArrow() {
		if(controllPlayer==controlPlayers.player1) {controlledPlayer = player1;}
		else {controlledPlayer = player2;}

		controller.Update();
		Vector3 move = controller.QueryLeftJoystickVector().ZeroY();
		if(enableControl==false || (move.x==0f&&move.z==0f)) {arrow.SetActive(false);}
		else{arrow.SetActive(true);}
		
		float signedAngle = Mathf.Atan2(move.x, move.z) * Mathf.Rad2Deg;
		
		arrow.transform.rotation = controlledPlayer.transform.rotation * Quaternion.AngleAxis(signedAngle, Vector3.up);
		Vector3 direction = arrow.transform.rotation.GetForward();
		float scale = 0.7f;
		arrow.transform.position = new Vector3(controlledPlayer.transform.position.x, arrow.transform.position.y, controlledPlayer.transform.position.z) + direction*scale;
	}


	private void Control() {
		AssignHistory(controlledPlayer);

		if(enableControl && controllPlayer==controlPlayers.player1) {ControlInputTrajectory(rootSeries);};

		//Interpolate Timeseries
		rootSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		rootSeries2.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		gtRootSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		gtRootSeries2.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);

		headSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		leftHandSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		rightHandSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		leftLegSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		rightLegSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);

		headSeries2.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		leftHandSeries2.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		rightHandSeries2.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		leftLegSeries2.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);
		rightLegSeries2.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);

		controlSeries.Interpolate(timeSeries.Pivot, timeSeries.Samples.Length);

		//Update Past
		rootSeries.Increment(0, timeSeries.Samples.Length-1);
		rootSeries2.Increment(0, timeSeries.Samples.Length-1);
		gtRootSeries.Increment(0, timeSeries.Samples.Length-1);
		gtRootSeries2.Increment(0, timeSeries.Samples.Length-1);

		controlSeries.Increment(0, timeSeries.Samples.Length-1);

		// joint history
		headSeries.Increment(0, timeSeries.Samples.Length-1);
		leftHandSeries.Increment(0, timeSeries.Samples.Length-1);
		rightHandSeries.Increment(0, timeSeries.Samples.Length-1);
		leftLegSeries.Increment(0, timeSeries.Samples.Length-1);
		rightLegSeries.Increment(0, timeSeries.Samples.Length-1);

		headSeries2.Increment(0, timeSeries.Samples.Length-1);
		leftHandSeries2.Increment(0, timeSeries.Samples.Length-1);
		rightHandSeries2.Increment(0, timeSeries.Samples.Length-1);
		leftLegSeries2.Increment(0, timeSeries.Samples.Length-1);
		rightLegSeries2.Increment(0, timeSeries.Samples.Length-1);

		// contact
		contactSeries.Increment(0, timeSeries.Samples.Length-1);
		contactSeries2.Increment(0, timeSeries.Samples.Length-1);
	}

	private Matrix4x4 ControlInputRoot(Vector3 offset) {
		controller.Update();

		Vector3 move = controller.QueryLeftJoystickVector().ZeroY();
		float unsignedAngle = Vector3.Angle(Vector3.forward, move);
        Vector3 crossProduct = Vector3.Cross(Vector3.forward, move);
        float signedAngle = unsignedAngle * Mathf.Sign(crossProduct.y);
		if(move.x==0f&&move.z==0f) {signedAngle=0f;}

		Matrix4x4 originalRoot = controlledPlayer.GetRoot().GetWorldMatrix(true);
		Matrix4x4 updatedRoot =  controlledPlayer.GetRoot().GetWorldMatrix(true) * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
		float magnitude = (updatedRoot.GetPosition()-originalRoot.GetPosition()).magnitude;
		Vector3 originalForward = (updatedRoot.GetPosition()-originalRoot.GetPosition()).normalized;
		Vector3 rotatedForward = (Quaternion.LookRotation(originalForward) * Quaternion.AngleAxis(signedAngle, Vector3.up)).GetForward().normalized;
		Matrix4x4 controlledRoot = Matrix4x4.TRS(originalRoot.GetPosition()+rotatedForward*magnitude, updatedRoot.rotation, Vector3.one);
		
		return controlledRoot;
		// return updatedRoot;
	}
	private void ControlInputTrajectory(RootSeries controlledSeries) {
		controller.Update();

		Vector3 move = controller.QueryLeftJoystickVector().ZeroY();
		float unsignedAngle = Vector3.Angle(Vector3.forward, move);
        Vector3 crossProduct = Vector3.Cross(Vector3.forward, move);
        float signedAngle = unsignedAngle * Mathf.Sign(crossProduct.y);
		if(move.x==0f&&move.z==0f) {signedAngle=0f;}

		// print("angle:"+signedAngle.ToString());

		for(int i=0; i<timeSeries.Samples.Length; i++) {
			controlSeries.SetTransformation(i, controlledSeries.Transformations[i]);
		}
		float scale = 0.05f;
		
		//Trajectory
		if(signedAngle!=0f) {
			for(int i=timeSeries.Pivot-1; i<timeSeries.Samples.Length-1; i++) {
				float magnitude = (controlledSeries.GetPosition(i+1)-controlledSeries.GetPosition(i)).magnitude;
				Vector3 originalForward = (controlledSeries.GetPosition(i+1)-controlledSeries.GetPosition(i)).normalized;
				Vector3 rotatedForward = (Quaternion.LookRotation(originalForward) * Quaternion.AngleAxis(signedAngle*scale*(i+1-timeSeries.Pivot), Vector3.up)).GetForward();
				controlSeries.SetPosition(i+1, controlSeries.GetPosition(i)+magnitude*rotatedForward);
			}
		}
	}

	private void CreateHistoryTrajectory() {
		List<Vector3> history = new List<Vector3>();
		historySeries.Add(history);
	}

	private void InitializeSeries(){
		timeSeries = new TimeSeries(6, 6, 1f, 1f, 5);

		rootSeries = new RootSeries(timeSeries, transform);
		headSeries = new RootSeries(timeSeries, transform);
		leftHandSeries = new RootSeries(timeSeries, transform);
		rightHandSeries = new RootSeries(timeSeries, transform);
		leftLegSeries = new RootSeries(timeSeries, transform);
		rightLegSeries = new RootSeries(timeSeries, transform);
		controlSeries = new RootSeries(timeSeries, transform);

		rootSeries2 = new RootSeries(timeSeries, transform);
		headSeries2 = new RootSeries(timeSeries, transform);
		leftHandSeries2 = new RootSeries(timeSeries, transform);
		rightHandSeries2 = new RootSeries(timeSeries, transform);
		leftLegSeries2 = new RootSeries(timeSeries, transform);
		rightLegSeries2 = new RootSeries(timeSeries, transform);

		gtRootSeries = new RootSeries(timeSeries, transform);
		gtRootSeries2 = new RootSeries(timeSeries, transform);

		testSeries = new RootSeries(timeSeries, transform);

		contactSeries = new ContactSeries(timeSeries, "Left Toe", "Right Toe");
		contactSeries2 = new ContactSeries(timeSeries, "Left Toe", "Right Toe");
	}

	private void ProcessFootContact(UltimateIK.Model ik, float contact) {
		ik.Activation = UltimateIK.ACTIVATION.Constant;
		ik.Objectives.First().SetTarget(Vector3.Lerp(ik.Objectives[0].TargetPosition, ik.Bones.Last().Transform.position, 1f-contact));
		ik.Objectives.First().SetTarget(ik.Bones.Last().Transform.rotation);
		ik.Iterations = 50;
		ik.Solve();
	}

	float DetectFootContact(UltimateIK.Model ik, float threshold_min, float threshold_max) {
        float y = ik.Bones.Last().Transform.position.y;
		float contact = 0f;
		if (y > threshold_max) {
            contact = 0f;
        }
        else if (y <= threshold_max && y > threshold_min) {
            float progress = Mathf.Clamp01(1 - y / threshold_max); 
            contact = Mathf.Pow(progress, 2); 
        }
        else if (y <= threshold_min) {
            contact = 1f;
        }

		return contact;
    }

	private void ProcessHandContact(UltimateIK.Model ik, Transform targetHand, float contact) {
		ik.Activation = UltimateIK.ACTIVATION.Constant;

		Vector3 targetPosition = targetHand.position; // + 0.15f * targetHand.forward; // - 0.3f * targetHand.up;
		ik.Objectives.First().SetTarget(Vector3.Lerp(targetPosition, ik.Bones.Last().Transform.position, 1f-contact));
		ik.Objectives.First().SetTarget(ik.Bones.Last().Transform.rotation);
		// ik.Objectives.First().SetTarget(targetPosition);

		ik.Iterations = 50;
		ik.Solve();
	}

	private float DetectHandContact(UltimateIK.Model ik, Vector3 targetHandPosition, float threshold_min, float threshold_max) {
		float distance = (ik.Bones.Last().Transform.position - targetHandPosition).magnitude;
		float contact = 0f;
		if (distance > threshold_max) {
            contact = 0f;
        }
        else if (distance <= threshold_max && distance > threshold_min) {
            // float progress = Mathf.Clamp01(1 - distance / threshold_max); 
            // contact = 1f - Mathf.Pow(progress, 2); 
			contact = (distance-threshold_min) / (threshold_max-threshold_min);
        }
        else if (distance <= threshold_min) {
            contact = 1f;
        }

		return contact;
	}

	private void AssignHistory(Actor actor) {
		if(historySeries.Count>0) {historySeries[historySeries.Count-1].Add(actor.transform.position);}
	}

    private void CorrectTwist(Actor character) {
        for(int i=0; i<character.Bones.Length; i++) {
            if(character.Bones[i].Childs.Length == 1) {
                Vector3 position = character.Bones[i].Transform.position;
                Quaternion rotation = character.Bones[i].Transform.rotation;
                Vector3 childPosition = character.Bones[i].GetChild(0).Transform.position;
                Quaternion childRotation = character.Bones[i].GetChild(0).Transform.rotation;
                // Vector3 aligned = (position - childPosition).normalized;
				Vector3 aligned = (childPosition - position).normalized;
                character.Bones[i].Transform.rotation = Quaternion.FromToRotation(rotation.GetUp(), aligned) * rotation;
                character.Bones[i].GetChild(0).Transform.position = childPosition;
                character.Bones[i].GetChild(0).Transform.rotation = childRotation;
            }
        }
    }

	private void ProcessFootIK(UltimateIK.Model ik, float contact) {
		ik.Activation = UltimateIK.ACTIVATION.Constant;
		ik.Objectives.First().SetTarget(Vector3.Lerp(ik.Objectives[0].TargetPosition, ik.Bones.Last().Transform.position, 1f-contact));
		ik.Objectives.First().SetTarget(ik.Bones.Last().Transform.rotation);
		ik.Iterations = 50;
		ik.Solve();
	}

	protected override void OnGUIDerived() {
	}

	protected override void OnRenderObjectDerived() {
		Color blueSkeleton = new Color(0.15f, 0.50f, 0.73f, 1f);
		Color blue = UltiDraw.Blue.Opacity(0.5f);
		Color redSkeleton = new Color(0.74f, 0.24f, 0.33f, 1f);
		Color red = UltiDraw.Red.Opacity(0.5f);
		Color orange = UltiDraw.Orange;//.Opacity(0.5f);
		Color green = UltiDraw.Green;//.Opacity(0.5f);
		Color purple = UltiDraw.Purple;//.Opacity(0.5f);
		Color cyan = UltiDraw.Cyan;//.Opacity(0.5f);
		Color mustard = UltiDraw.Mustard;//.Opacity(0.5f);
		List<Color> colors = new List<Color>() {
			new Color(0.15f, 0.5f, 0.73f, 0.5f),
			new Color(0.74f, 0.24f, 0.33f, 0.5f),
			new Color(0.15f, 0.73f, 0.23f, 0.5f),
		};

		if(drawRootSeries!=DrawRootSeries.None) {
			if((drawRootSeries & DrawRootSeries.player1)!=0) {
				rootSeries.Draw(GetCamera(), blueSkeleton, true, drawHalf: 2);
			}
			if((drawRootSeries & DrawRootSeries.player2)!=0) {
				rootSeries2.Draw(GetCamera(), redSkeleton, true, drawHalf: 2);
			}
		}

		if(drawJointSeries!=DrawJointSeries.None && NeuralNetwork.countFrame>NeuralNetwork.initHistory) {
			Color jointTrajColor = orange;
			if(enableGtIntention) jointTrajColor = green;
			if((drawJointSeries & DrawJointSeries.player1)!=0) {
				// headSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2, local: player2.GetRoot().GetWorldMatrix(true));
				// leftHandSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2, local: player2.GetRoot().GetWorldMatrix(true));
				// leftLegSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2, local: player2.GetRoot().GetWorldMatrix(true));
				// rightHandSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2, local: player2.GetRoot().GetWorldMatrix(true));
				// rightLegSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2, local: player2.GetRoot().GetWorldMatrix(true));

				headSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				leftHandSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				leftLegSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				rightHandSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				rightLegSeries.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
			}

			if((drawJointSeries & DrawJointSeries.player2)!=0) {
				headSeries2.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				leftHandSeries2.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				leftLegSeries2.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				rightHandSeries2.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
				rightLegSeries2.Draw(GetCamera(), jointTrajColor, false, drawHalf: 2);
			}
		}

		if(enableControl && controllPlayer==controlPlayers.player1) {
			rootSeries.DrawWithOffset(GetCamera(), blueSkeleton, drawHalf: 2, offset: 10f);
			if(arrow.activeSelf==true) {
				controlSeries.Draw(GetCamera(), UltiDraw.Orange.Opacity(0.5f), true, drawHalf: 2);
				controlSeries.DrawWithOffset(GetCamera(), UltiDraw.Orange.Opacity(0.5f), true, drawHalf: 2, offset: 10f);
			}
		}
		else {
			rootSeries.DrawWithOffset(GetCamera(), blueSkeleton, drawHalf: 0, offset: 10f);
		}

		if(enableTestPlayer) {
			testPlayer.gameObject.SetActive(true);
			testSeries.Draw(GetCamera(), UltiDraw.Black.Opacity(0.5f), true, drawHalf: 0);
		}
		else {testPlayer.gameObject.SetActive(false);}

		if(NeuralNetwork.enableSocket==false) {
			gtPlayer1.gameObject.SetActive(false);
			gtPlayer2.gameObject.SetActive(false);
		}

		if(drawHistories) {
			UltiDraw.Begin();
			int step=5;
			for(int h=0; h<historySeries.Count; h++) {
				for(int p=0; p<historySeries[h].Count; p+=step) {
					UltiDraw.DrawCircle(historySeries[h][p], size: 0.05f, color: colors[h]);
					if(p>step) {UltiDraw.DrawLine(historySeries[h][p], historySeries[h][p-step], Vector3.up, 0.03f, colors[h]);}
				}
			}
			UltiDraw.End();
		}

		// if(drawSamples && NeuralNetwork.countFrame>NeuralNetwork.initHistory && NeuralNetwork.enableSocket==true) {
		if(drawSamples) {
			samplesObject.SetActive(true);
			UltiDraw.Begin();
			for(int s=1; s<states.Count; s++) {
				Actor drawActor = samplesObject.transform.Find(s.ToString()).GetComponent<Actor>();
				states[s].DrawSamples(drawActor, camera, drawPlayer1:((drawJointSeries & DrawJointSeries.player1)!=0), drawPlayer2:((drawJointSeries & DrawJointSeries.player2)!=0));
			}
			UltiDraw.End();
		}
		else {samplesObject.SetActive(false);}

		// contact
		if(drawContact) {
			UltiDraw.Begin();
			float Threshold = 0.05f;
			Quaternion rot = player1.GetBoneTransformation("LeftToeBase").GetRotation();
			Vector3 pos = player1.GetBoneTransformation("LeftToeBase").GetPosition();
			UltiDraw.DrawCube(pos, rot, 0.025f, UltiDraw.Black);
			UltiDraw.DrawWireSphere(pos, rot, 2f*Threshold, green.Opacity(0.25f));
			if(contactSeries.Values[timeSeries.GetPivot().Index][0] == 1f) {
				UltiDraw.DrawSphere(pos, rot, 2f*Threshold, green);
			} else {
				UltiDraw.DrawSphere(pos, rot, 2f*Threshold, green.Opacity(0.125f));
			}

			rot = player1.GetBoneTransformation("RightToeBase").GetRotation();
			pos = player1.GetBoneTransformation("RightToeBase").GetPosition();
			UltiDraw.DrawCube(pos, rot, 0.025f, UltiDraw.Black);
			UltiDraw.DrawWireSphere(pos, rot, 2f*Threshold, green.Opacity(0.25f));
			if(contactSeries.Values[timeSeries.GetPivot().Index][1] == 1f) {
				UltiDraw.DrawSphere(pos, rot, 2f*Threshold, green);
			} else {
				UltiDraw.DrawSphere(pos, rot, 2f*Threshold, green.Opacity(0.125f));
			}
			UltiDraw.End();
		}
	}

	public void SetFramerate(float rate) {
		framerate = rate;
	}

	protected override void Feed2() {
	}

	protected override void Read2() {
	}
}