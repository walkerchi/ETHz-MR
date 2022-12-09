﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class KeyboardCommand : MonoBehaviour
{
    public GameObject debugText;

    ResearchModeVideoStream rm;

    void Start()
    {
        rm = GetComponent<ResearchModeVideoStream>();
    }

    // Update is called once per frame
    void Update()
    {
/* #if ENABLE_WINMD_SUPPORT
        if (Input.GetKeyDown(KeyCode.Space))
        {
            rm.SendRawData();
        }
#endif */
    }

    public void Exit() {
        Application.Quit();
    }

    public void ToggleDebugText() {
        if (debugText != null) debugText.SetActive(!debugText.activeInHierarchy);
    }
}
