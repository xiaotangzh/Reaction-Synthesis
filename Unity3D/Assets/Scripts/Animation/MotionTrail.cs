﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Reflection;
using Accord.Math;

public class MotionTrail : MonoBehaviour {

	public int TrailLength = 10;
	public float TimeDifference = 0.1f;

	private GameObject Prototype;
	private Queue<GameObject> Instances;

	private System.DateTime Timestamp;
	public float Transparency = 0.25f;
	public bool UseMouseClick = false;
	public bool DynamicTransparencyToTarget = false;
	public Actor actor;
	private Actor target;
	public GameObject Target;
	public NeuralAnimation controller;
	private float distance_to_target = 0;

	void Start() {
		Instances = new Queue<GameObject>();
		Timestamp = Utility.GetTimestamp();
		Prototype = CreatePrototype();
	}

	void OnDestroy() {
		if(Prototype != null) {
			Utility.Destroy(Prototype);
		}
	}

	void Update() {
		distance_to_target = ComputeDistanceToTarget(actor, target);
		if(DynamicTransparencyToTarget==true) {Transparency = ComputeTransparency(distance_to_target);}

		if(UseMouseClick==false && Utility.GetElapsedTime(Timestamp) >= TimeDifference) {
			Timestamp = Utility.GetTimestamp();
			while(Instances.Count >= TrailLength) {
				Utility.Destroy(Instances.Dequeue());
			}
			Instances.Enqueue(CreateInstance());
		}

		if(UseMouseClick==true && Input.GetMouseButtonDown(0) && Transparency>=0.13f) {
			while(Instances.Count >= TrailLength) {
				Utility.Destroy(Instances.Dequeue());
			}
			Instances.Enqueue(CreateInstance());
		}
	}

	void OnRenderObject() {
		UltiDraw.Begin();
		int index = 0;
		//GameObject previous = null;
		foreach(GameObject instance in Instances) {
			index += 1;
			// instance.GetComponent<Transparency>().SetTransparency(Transparency);
			instance.GetComponent<Transparency>().SetTransparency();
			if(index > 1) {
				// UltiDraw.DrawSphere(instance.transform.position, Quaternion.identity, 0.025f, UltiDraw.Magenta.Opacity(0.8f));
			}
			//previous = instance;
		}
		UltiDraw.End();
	}

	void OnGUI() {
/* 		GUI.color = Color.black;
		if(GUI.Button(Utility.GetGUIRect(0.025f, 0.125f, 0.1f, 0.025f), "Trail +")) {
			TrailLength += 1;
		}
		if(GUI.Button(Utility.GetGUIRect(0.025f, 0.15f, 0.1f, 0.025f), "Trail -")) {
			TrailLength -= 1;
		}
		if(GUI.Button(Utility.GetGUIRect(0.025f, 0.175f, 0.1f, 0.025f), "Time +")) {
			TimeDifference += 0.1f;
		}
		if(GUI.Button(Utility.GetGUIRect(0.025f, 0.2f, 0.1f, 0.025f), "Time -")) {
			TimeDifference -= 0.1f;
		}
		GUI.Label(Utility.GetGUIRect(0.025f, 0.125f, 0.1f, 0.025f), "Trail: " + TrailLength);
		GUI.Label(Utility.GetGUIRect(0.025f, 0.15f, 0.1f, 0.025f), "Time: " + TimeDifference); */
	}

	private GameObject CreatePrototype() {
		GameObject instance = Instantiate(gameObject);
		// GameObject instance = Instantiate(Target); //todo
		instance.name = "Prototype";
		instance.SetActive(false);
		instance.hideFlags = HideFlags.HideInHierarchy;
		Cleanup(instance.transform);
		instance.AddComponent<Transparency>();
		return instance;
	}

	private GameObject CreateInstance() {
		GameObject instance = Instantiate(Prototype);
		instance.name = name + " (Motion Trail)";
		instance.SetActive(true);
		instance.GetComponent<Transparency>().Alpha = Transparency; //todo
		Copy(gameObject.transform, instance.transform);
		return instance;
	}

	private void Cleanup(Transform t) {
		foreach(Component c in t.GetComponents<Component>()) {
			if(!(c is Transform)) {
				if(c is Renderer) {
					Renderer r = (Renderer)c;
					if(!r.material.HasProperty("_Color")) {
						// Utility.Destroy(c);
					}
				} else {
					Utility.Destroy(c);
				}
			}
		}
		// for(int i=0; i<t.childCount; i++) {
		// 	Cleanup(t.GetChild(i));
		// }
	}

	private void Copy(Transform original, Transform instance) {
		// instance.localPosition = original.localPosition;
		// instance.localRotation = original.localRotation;
		// instance.localScale = original.localScale;
		instance.position = original.position;
		instance.rotation = original.rotation;
		for(int i=0; i<original.childCount; i++) {
			Copy(original.GetChild(i), instance.GetChild(i));
		}
	}

	private float ComputeDistanceToTarget(Actor actor, Actor target) {
		return (actor.Bones[0].Transform.position - target.Bones[0].Transform.position).magnitude;
	}
	private float ComputeTransparency(float distance) {
		if(distance>=1f) return 0;
		else return (1f - distance)/1f * 0.4f;
	}

}
