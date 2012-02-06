// -*- c-basic-offset: 4; related-file-name: "../include/click/notifier.hh" -*-
/*
 * notifier.{cc,hh} -- activity notification
 * Eddie Kohler
 *
 * Copyright (c) 2002 International Computer Science Institute
 * Copyright (c) 2004-2005 Regents of the University of California
 * Copyright (c) 2008 Meraki, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, subject to the conditions
 * listed in the Click LICENSE file. These conditions include: you must
 * preserve this copyright notice, and you cannot mention the copyright
 * holders in advertising related to the Software without their permission.
 * The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
 * notice is a summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */

#include <click/config.h>
#if __GNUC__
# pragma implementation "click/notifier.hh"
#endif
#include <click/notifier.hh>
#include <click/router.hh>
#include <click/element.hh>
#include <click/routervisitor.hh>
#include <click/straccum.hh>
#include <click/bitvector.hh>
CLICK_DECLS

// should be const, but we need to explicitly initialize it
atomic_uint32_t NotifierSignal::static_value;
const char Notifier::EMPTY_NOTIFIER[] = "empty";
const char Notifier::FULL_NOTIFIER[] = "full";

/** @file notifier.hh
 * @brief Support for activity signals.
 */

/** @class NotifierSignal
 * @brief An activity signal.
 *
 * Activity signals in Click let one element determine whether another element
 * is active.  For example, consider an element @e X pulling from a @e Queue.
 * If the @e Queue is empty, there's no point in @e X trying to pull from it.
 * Thus, the @e Queue has an activity signal that's active when it contains
 * packets and inactive when it's empty.  @e X can check the activity signal
 * before pulling, and do something else if it's inactive.  Combined with the
 * sleep/wakeup functionality of ActiveNotifier, this can greatly reduce CPU
 * load due to polling.
 *
 * A "basic activity signal" is essentially a bit that's either on or off.
 * When it's on, the signal is active.  NotifierSignal can represent @e
 * derived activity signals as well.  A derived signal combines information
 * about @e N basic signals using the following invariant: If any of the basic
 * signals is active, then the derived signal is also active.  There are no
 * other guarantees; in particular, the derived signal might be active even if
 * @e none of the basic signals are active.
 *
 * Click elements construct NotifierSignal objects in four ways:
 *
 *  - idle_signal() returns a signal that's never active.
 *  - busy_signal() returns a signal that's always active.
 *  - Router::new_notifier_signal() creates a new basic signal.  This method
 *    should be preferred to NotifierSignal's own constructors.
 *  - operator+(NotifierSignal, const NotifierSignal &) creates a derived signal.
 */

/** @class Notifier
 * @brief A basic activity signal and notification provider.
 *
 * The Notifier class represents a basic activity signal associated with an
 * element.  Elements that contain a Notifier object will override
 * Element::cast() or Element::port_cast() to return that Notifier when given
 * the proper name.  This lets other parts of the configuration find the
 * Notifiers.  See upstream_empty_signal() and downstream_full_signal().
 *
 * The ActiveNotifier class, which derives from Notifier, can wake up clients
 * when its activity signal becomes active.
 */

/** @class ActiveNotifier
 * @brief A basic activity signal and notification provider that can
 * reschedule any dependent Task objects.
 *
 * ActiveNotifier, whose base class is Notifier, combines a basic activity
 * signal with the ability to wake up any dependent Task objects when that
 * signal becomes active.  Notifier clients are called @e listeners.  Each
 * listener corresponds to a Task object.  The listener generally goes to
 * sleep -- i.e., becomes unscheduled -- when it runs out of work and the
 * corresponding activity signal is inactive.  The ActiveNotifier class will
 * wake up the listener when it becomes active by rescheduling the relevant
 * Task.
 *
 * Elements that contain ActiveNotifier objects will generally override
 * Element::cast() or Element::port_cast(), allowing other parts of the
 * configuration to find the Notifiers.
 */


/** @brief Initialize the NotifierSignal implementation.
 *
 * This function must be called before NotifierSignal functionality is used.
 * It is safe to call it multiple times.
 *
 * @note Elements don't need to worry about static_initialize(); Click drivers
 * have already called it for you.
 */
void
NotifierSignal::static_initialize()
{
    static_value = true_mask | overderived_mask;
}

NotifierSignal &
NotifierSignal::operator+=(const NotifierSignal &x)
{
    // preserve busy_signal(); adding other incompatible signals
    // leads to overderived_signal()
    if (busy() || x.idle())
	/* do nothing */;
    else if (idle() || x.busy())
	*this = x;
    else if (_mask && x._mask && _v.v1 == x._v.v1)
	_mask |= x._mask;
    else if (x._mask)
	hard_derive_one(x._v.v1, x._mask);
    else if (this != &x)
	for (vmpair *vm = x._v.vm; vm->mask; ++vm)
	    hard_derive_one(vm->value, vm->mask);

    return *this;
}

void
NotifierSignal::hard_assign_vm(const NotifierSignal &x)
{
    size_t n = 0;
    for (vmpair *vm = x._v.vm; vm->mask; ++vm)
	++n;
    if (likely((_v.vm = new vmpair[n + 1])))
	memcpy(_v.vm, x._v.vm, sizeof(vmpair) * (n + 1));
    else
	*this = overderived_signal();
}

void
NotifierSignal::hard_derive_one(atomic_uint32_t *value, uint32_t mask)
{
    if (unlikely(_mask)) {
	if (busy())
	    return;
	if (_v.v1 == value) {
	    _mask |= mask;
	    return;
	}
	vmpair *vmp;
	if (unlikely(!(vmp = new vmpair[2]))) {
	    *this = overderived_signal();
	    return;
	}
	vmp[0].value = _v.v1;
	vmp[0].mask = _mask;
	vmp[1].mask = 0;
	_v.vm = vmp;
	_mask = 0;
    }

    size_t n, i;
    vmpair *vmp;
    for (i = 0, vmp = _v.vm; vmp->mask && vmp->value < value; ++i, ++vmp)
	/* do nothing */;
    if (vmp->mask && vmp->value == value) {
	vmp->mask |= mask;
	return;
    }
    for (n = i; vmp->mask; ++n, ++vmp)
	/* do nothing */;

    if (unlikely(!(vmp = new vmpair[n + 2]))) {
	*this = overderived_signal();
	return;
    }
    memcpy(vmp, _v.vm, sizeof(vmpair) * i);
    memcpy(vmp + i + 1, _v.vm + i, sizeof(vmpair) * (n + 1 - i));
    vmp[i].value = value;
    vmp[i].mask = mask;
    delete[] _v.vm;
    _v.vm = vmp;
}

bool
NotifierSignal::hard_equals(const vmpair *a, const vmpair *b)
{
    while (a->mask && a->mask == b->mask && a->value == b->value)
	++a, ++b;
    return !a->mask && a->mask == b->mask;
}

String
NotifierSignal::unparse(Router *router) const
{
    if (!_mask) {
	StringAccum sa;
	for (vmpair *vm = _v.vm; vm->mask; ++vm)
	    sa << (vm == _v.vm ? "" : "+")
	       << NotifierSignal(vm->value, vm->mask).unparse(router);
	return sa.take_string();
    }

    char buf[80];
    int pos;
    String s;
    if (_v.v1 == &static_value) {
	if (_mask == true_mask)
	    return "busy*";
	else if (_mask == false_mask)
	    return "idle";
	else if (_mask == overderived_mask)
	    return "overderived*";
	else if (_mask == uninitialized_mask)
	    return "uninitialized";
	else
	    pos = sprintf(buf, "internal/");
    } else if (router && (s = router->notifier_signal_name(_v.v1)) >= 0) {
	pos = sprintf(buf, "%.52s/", s.c_str());
    } else
	pos = sprintf(buf, "@%p/", _v.v1);
    sprintf(buf + pos, active() ? "%x:%x*" : "%x:%x", _mask, (*_v.v1) & _mask);
    return String(buf);
}


/** @brief Destruct a Notifier. */
Notifier::~Notifier()
{
}

/** @brief Called to register a listener with this Notifier.
 * @param task the listener's Task
 *
 * This notifier should register @a task as a listener, if appropriate.
 * Later, when the signal is activated, the Notifier should reschedule @a task
 * along with the other listeners.  Not all types of Notifier need to provide
 * this functionality, however.  The default implementation does nothing.
 */
int
Notifier::add_listener(Task* task)
{
    (void) task;
    return 0;
}

/** @brief Called to unregister a listener with this Notifier.
 * @param task the listener's Task
 *
 * Undoes the effect of any prior add_listener(@a task).  Should do nothing if
 * @a task was never added.  The default implementation does nothing.
 */
void
Notifier::remove_listener(Task* task)
{
    (void) task;
}

/** @brief Called to register a dependent signal with this Notifier.
 * @param signal the dependent signal
 *
 * This notifier should register @a signal as a dependent signal, if
 * appropriate.  Later, when this notifier's signal is activated, it should go
 * ahead and activate @a signal as well.  Not all types of Notifier need to
 * provide this functionality.  The default implementation does nothing.
 */
int
Notifier::add_dependent_signal(NotifierSignal* signal)
{
    (void) signal;
    return 0;
}

/** @brief Initialize the associated NotifierSignal, if necessary.
 * @param name signal name
 * @param r associated router
 *
 * Initialize the Notifier's associated NotifierSignal by calling @a r's
 * Router::new_notifier_signal() method, obtaining a new basic activity
 * signal.  Does nothing if the signal is already initialized.
 */
int
Notifier::initialize(const char *name, Router *r)
{
    if (!_signal.initialized())
	return r->new_notifier_signal(name, _signal);
    else
	return 0;
}


/** @brief Construct an ActiveNotifier.
 * @param op controls notifier path search
 *
 * Constructs an ActiveNotifier object, analogous to the
 * Notifier::Notifier(SearchOp) constructor.  (See that constructor for more
 * information on @a op.)
 */
ActiveNotifier::ActiveNotifier(SearchOp op)
    : Notifier(op), _listener1(0), _listeners(0)
{
}

/** @brief Destroy an ActiveNotifier. */
ActiveNotifier::~ActiveNotifier()
{
    delete[] _listeners;
}

int
ActiveNotifier::listener_change(void *what, int where, bool add)
{
    int n = 0, x;
    task_or_signal_t *tos, *ntos, *otos;

    // common case
    if (!_listener1 && !_listeners && where == 0 && add) {
	_listener1 = (Task *) what;
	return 0;
    }

    for (tos = _listeners, x = 0; tos && x < 2; tos++)
	tos->v ? n++ : x++;
    if (_listener1)
	n++;

    if (!(ntos = new task_or_signal_t[n + 2 + add])) {
      memory_error:
	delete[] ntos;
	click_chatter("out of memory in Notifier!");
	return -ENOMEM;
    }

    otos = ntos;
    if (!_listeners) {
	// handles both the case of _listener1 != 0 and _listener1 == 0
	if (!(_listeners = new task_or_signal_t[3]))
	    goto memory_error;
	_listeners[0].t = _listener1;
	_listeners[1].v = _listeners[2].v = 0;
    }
    for (tos = _listeners, x = 0; x < 2; tos++)
	if (tos->v && (add || tos->v != what)) {
	    (otos++)->v = tos->v;
	    if (tos->v == what)
		add = false;
	} else if (!tos->v) {
	    if (add && where == x)
		(otos++)->v = what;
	    (otos++)->v = 0;
	    x++;
	}
    assert(otos - ntos <= n + 2 + add);

    delete[] _listeners;
    if (!ntos[0].v && !ntos[1].v) {
	_listeners = 0;
	_listener1 = 0;
	delete[] ntos;
    } else if (ntos[0].v && !ntos[1].v && !ntos[2].v) {
	_listeners = 0;
	_listener1 = ntos[0].t;
	delete[] ntos;
    } else {
	_listeners = ntos;
	_listener1 = 0;
    }
    return 0;
}

/** @brief Add a listener to this notifier.
 * @param task the listener to add
 *
 * Adds @a task to this notifier's listener list (the clients interested in
 * notification).  Whenever the ActiveNotifier activates its signal, @a task
 * will be rescheduled.
 */
int
ActiveNotifier::add_listener(Task* task)
{
    return listener_change(task, 0, true);
}

/** @brief Remove a listener from this notifier.
 * @param task the listener to remove
 *
 * Removes @a task from this notifier's listener list (the clients interested
 * in notification).  @a task will not be rescheduled when the Notifier is
 * activated.
 */
void
ActiveNotifier::remove_listener(Task* task)
{
    listener_change(task, 0, false);
}

/** @brief Add a dependent signal to this Notifier.
 * @param signal the dependent signal
 *
 * Adds @a signal as a dependent signal to this notifier.  Whenever the
 * ActiveNotifier activates its signal, @a signal will be activated as well.
 */
int
ActiveNotifier::add_dependent_signal(NotifierSignal* signal)
{
    return listener_change(signal, 1, true);
}

/** @brief Return the listener list.
 * @param[out] v collects listener tasks
 *
 * Pushes all listener Task objects onto the end of @a v.
 */
void
ActiveNotifier::listeners(Vector<Task*>& v) const
{
    if (_listener1)
	v.push_back(_listener1);
    else if (_listeners)
	for (task_or_signal_t* l = _listeners; l->t; l++)
	    v.push_back(l->t);
}


namespace {

class NotifierRouterVisitor : public RouterVisitor { public:
    NotifierRouterVisitor(const char* name);
    bool visit(Element *e, bool isoutput, int port,
	       Element *from_e, int from_port, int distance);
    Vector<Notifier*> _notifiers;
    NotifierSignal _signal;
    bool _pass2;
    bool _need_pass2;
    const char* _name;
};

NotifierRouterVisitor::NotifierRouterVisitor(const char* name)
    : _signal(NotifierSignal::idle_signal()),
      _pass2(false), _need_pass2(false), _name(name)
{
}

bool
NotifierRouterVisitor::visit(Element* e, bool isoutput, int port,
			     Element *, int, int)
{
    if (Notifier* n = (Notifier*) (e->port_cast(isoutput, port, _name))) {
	if (find(_notifiers.begin(), _notifiers.end(), n) == _notifiers.end())
	    _notifiers.push_back(n);
	if (!n->signal().initialized())
	    n->initialize(_name, e->router());
	_signal += n->signal();
	Notifier::SearchOp search_op = n->search_op();
	if (search_op == Notifier::SEARCH_CONTINUE_WAKE && !_pass2) {
	    _need_pass2 = true;
	    return false;
	} else
	    return search_op != Notifier::SEARCH_STOP;

    } else if (port >= 0) {
	Bitvector flow;
	if (e->port_active(isoutput, port)) {
	    // went from pull <-> push
	    _signal = NotifierSignal::busy_signal();
	    return false;
	} else if ((e->port_flow(isoutput, port, &flow), flow.zero())
		   && e->flag_value('S') != 0) {
	    // ran out of ports, but element might generate packets
	    _signal = NotifierSignal::busy_signal();
	    return false;
	} else
	    return true;

    } else
	return true;
}

}

/** @brief Calculate and return the NotifierSignal derived from all empty
 * notifiers upstream of element @a e's input @a port, and optionally register
 * @a task as a listener.
 * @param e an element
 * @param port the input port of @a e at which to start the upstream search
 * @param task Task to register as a listener, or null
 * @param dependent_notifier Notifier to register as dependent, or null
 *
 * Searches the configuration upstream of element @a e's input @a port for @e
 * empty @e notifiers.  These notifiers are associated with packet storage,
 * and should be true when packets are available (or likely to be available
 * quite soon), and false when they are not.  All notifiers found are combined
 * into a single derived signal.  Thus, if any of the base notifiers are
 * active, indicating that at least one packet is available upstream, the
 * derived signal will also be active.  Element @a e's code generally uses the
 * resulting signal to decide whether or not to reschedule itself.
 *
 * The returned signal is generally conservative, meaning that the signal
 * is true whenever a packet exists upstream, but the elements that provide
 * notification are responsible for ensuring this.
 *
 * If @a task is nonnull, then @a task becomes a listener for each located
 * notifier.  Thus, when a notifier becomes active (when packets become
 * available), @a task will be rescheduled.
 *
 * If @a dependent_notifier is null, then its signal is registered as a
 * <em>dependent signal</em> on each located upstream notifier.  When
 * an upstream notifier becomes active, @a dependent_notifier's signal is also
 * activated.
 *
 * <h3>Supporting upstream_empty_signal()</h3>
 *
 * Elements that have an empty notifier must override the Element::cast()
 * method.  When passed the @a name Notifier::EMPTY_NOTIFIER, this method
 * should return a pointer to the corresponding Notifier object.
 */
NotifierSignal
Notifier::upstream_empty_signal(Element* e, int port, Task* task, Notifier* dependent_notifier)
{
    NotifierRouterVisitor filter(EMPTY_NOTIFIER);
    int ok = e->router()->visit_upstream(e, port, &filter);

    NotifierSignal signal = filter._signal;

    // maybe run another pass
    if (ok >= 0 && signal != NotifierSignal() && filter._need_pass2) {
	filter._pass2 = true;
	ok = e->router()->visit_upstream(e, port, &filter);
    }

    // All bets are off if filter ran into a push output. That means there was
    // a regular Queue in the way (for example).
    if (ok < 0 || signal == NotifierSignal())
	return NotifierSignal();

    if (task)
	for (int i = 0; i < filter._notifiers.size(); i++)
	    filter._notifiers[i]->add_listener(task);
    if (dependent_notifier)
	for (int i = 0; i < filter._notifiers.size(); i++)
	    filter._notifiers[i]->add_dependent_signal(&dependent_notifier->_signal);

    return signal;
}

/** @brief Calculate and return the NotifierSignal derived from all full
 * notifiers downstream of element @a e's output @a port, and optionally
 * register @a task as a listener.
 * @param e an element
 * @param port the output port of @a e at which to start the downstream search
 * @param task Task to register as a listener, or null
 * @param dependent_notifier Notifier to register as dependent, or null
 *
 * Searches the configuration downstream of element @a e's output @a port for
 * @e full @e notifiers.  These notifiers are associated with packet storage,
 * and should be true when there is space for at least one packet, and false
 * when there is not.  All notifiers found are combined into a single derived
 * signal.  Thus, if any of the base notifiers are active, indicating that at
 * least one path has available space, the derived signal will also be active.
 * Element @a e's code generally uses the resulting signal to decide whether
 * or not to reschedule itself.
 *
 * If @a task is nonnull, then @a task becomes a listener for each located
 * notifier.  Thus, when a notifier becomes active (when space become
 * available), @a task will be rescheduled.
 *
 * If @a dependent_notifier is null, then its signal is registered as a
 * <em>dependent signal</em> on each located downstream notifier.  When
 * an downstream notifier becomes active, @a dependent_notifier's signal is
 * also activated.
 *
 * In current Click, the returned signal is conservative: if it's inactive,
 * then there is no space for packets downstream.
 *
 * <h3>Supporting downstream_full_signal()</h3>
 *
 * Elements that have a full notifier must override the Element::cast()
 * method.  When passed the @a name Notifier::FULL_NOTIFIER, this method
 * should return a pointer to the corresponding Notifier object.
 */
NotifierSignal
Notifier::downstream_full_signal(Element* e, int port, Task* task, Notifier* dependent_notifier)
{
    NotifierRouterVisitor filter(FULL_NOTIFIER);
    int ok = e->router()->visit_downstream(e, port, &filter);

    NotifierSignal signal = filter._signal;

    // maybe run another pass
    if (ok >= 0 && signal != NotifierSignal() && filter._need_pass2) {
	filter._pass2 = true;
	ok = e->router()->visit_downstream(e, port, &filter);
    }

    // All bets are off if filter ran into a pull input. That means there was
    // a regular Queue in the way (for example).
    if (ok < 0 || signal == NotifierSignal())
	return NotifierSignal();

    if (task)
	for (int i = 0; i < filter._notifiers.size(); i++)
	    filter._notifiers[i]->add_listener(task);
    if (dependent_notifier)
	for (int i = 0; i < filter._notifiers.size(); i++)
	    filter._notifiers[i]->add_dependent_signal(&dependent_notifier->_signal);

    return signal;
}

CLICK_ENDDECLS
