#!/usr/bin/env python3
"""Monitor GEE export tasks"""
import ee
ee.Initialize(project='skysensebarranquilla')

tasks = ee.data.getTaskList()
barranquilla_tasks = [t for t in tasks if 'Barranquilla' in t.get('description', '')]

if not barranquilla_tasks:
    print("No tasks found for Barranquilla")
else:
    print(f"{'ID':<10} {'STATE':<12} {'DESCRIPTION'}")
    print("-" * 80)
    for t in barranquilla_tasks:
        print(f"{t['id'][:8]:<10} {t['state']:<12} {t['description']}")
    
    # Count by state
    states = {}
    for t in barranquilla_tasks:
        states[t['state']] = states.get(t['state'], 0) + 1
    
    print("\nResumen:")
    for state, count in states.items():
        print(f"  {state}: {count}")
    
    completed = states.get('COMPLETED', 0)
    failed = states.get('FAILED', 0)
    running = states.get('RUNNING', 0) + states.get('READY', 0)
    total = len(barranquilla_tasks)
    
    print(f"\nProgreso: {completed}/{total} completados, {running} en proceso, {failed} fallidos")